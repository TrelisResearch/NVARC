"""
Validate all ARC-AGI-2 training tasks with Prolog-based batched parallel execution.

Run: uv run validate_prolog_all.py
     uv run validate_prolog_all.py --sample 10              # test on 10 tasks first
     uv run validate_prolog_all.py --max-concurrent 64      # max concurrent API calls
     uv run validate_prolog_all.py --resume                 # resume from checkpoint
"""
import argparse
import json
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from validate_prolog_task import (
    generate_input_recognizer,
    test_input_recognizer,
    generate_transform,
    test_transform_full,
    get_client,
    TASKS_FILE,
    SOLUTIONS_FILE,
    INPUT_PRICE,
    OUTPUT_PRICE,
)

RESULTS_FILE = Path(__file__).parent / "prolog_validation_results.jsonl"
SUMMARY_FILE = Path(__file__).parent / "prolog_validation_summary.json"

# Thread lock for file I/O
file_lock = threading.Lock()


def load_completed_tasks() -> set:
    """Load task IDs that have already been validated."""
    completed = set()
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE) as f:
            for line in f:
                if line.strip():
                    result = json.loads(line)
                    completed.add(result["task_id"])
    return completed


def save_result(result: dict):
    """Append a single result to the JSONL file (thread-safe)."""
    with file_lock:
        with open(RESULTS_FILE, "a") as f:
            f.write(json.dumps(result) + "\n")


def save_summary(stats: dict):
    """Save summary statistics (thread-safe)."""
    with file_lock:
        with open(SUMMARY_FILE, "w") as f:
            json.dump(stats, f, indent=2)


def run_recognizer_phase(
    task_ids: list,
    all_tasks: dict,
    all_solutions: dict,
    client,
    max_concurrent: int = 64,
) -> dict:
    """Generate and test input recognizers for all tasks in parallel pool."""
    results = {}
    total = len(task_ids)
    completed = [0]
    lock = threading.Lock()

    def generate_and_test_rec(task_id):
        task_data = all_tasks[task_id]
        test_inputs = [ex["input"] for ex in task_data.get("test", [])]

        rec_result = generate_input_recognizer(client, task_data)
        if rec_result["code"] is None:
            return task_id, {
                "success": False,
                "error": "No recognizer code parsed",
                "input_tokens": rec_result["input_tokens"],
                "output_tokens": rec_result["output_tokens"],
                "thinking_tokens": rec_result["thinking_tokens"],
            }

        test_result = test_input_recognizer(rec_result["code"], task_data, test_inputs)
        return task_id, {
            "success": test_result["success"],
            "code": rec_result["code"] if test_result["success"] else None,
            "train_passed": test_result["train_passed"],
            "train_total": test_result["train_total"],
            "test_passed": test_result["test_passed"],
            "test_total": test_result["test_total"],
            "specificity_passed": test_result["specificity_passed"],
            "errors": test_result.get("errors", []),
            "input_tokens": rec_result["input_tokens"],
            "output_tokens": rec_result["output_tokens"],
            "thinking_tokens": rec_result["thinking_tokens"],
        }

    print(f"\n=== RECOGNIZER PHASE ({total} tasks, {max_concurrent} concurrent) ===")

    with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
        futures = {executor.submit(generate_and_test_rec, task_id): task_id for task_id in task_ids}

        for future in as_completed(futures):
            task_id, result = future.result()
            results[task_id] = result

            with lock:
                completed[0] += 1
                status = "✓" if result["success"] else "✗"
                train = f"{result.get('train_passed', 0)}/{result.get('train_total', '?')}"
                test = f"{result.get('test_passed', 0)}/{result.get('test_total', '?')}"
                spec = "✓" if result.get("specificity_passed") else "✗"
                print(f"[{completed[0]}/{total}] {task_id}: {status} (train={train}, test={test}, spec={spec})")

    success_count = sum(1 for r in results.values() if r["success"])
    print(f"Recognizer phase complete: {success_count}/{total} successful")
    return results


def run_transform_phase(
    recognizers: dict,
    all_tasks: dict,
    all_solutions: dict,
    client,
    max_concurrent: int = 64,
    max_attempts: int = 4,
) -> dict:
    """Generate transform programs for tasks with valid recognizers, up to max_attempts each."""
    # Build pending dict: tasks with valid recognizers
    pending = {}
    for task_id, rec_result in recognizers.items():
        if rec_result["success"] and rec_result.get("code"):
            pending[task_id] = {
                "recognizer_code": rec_result["code"],
                "task_data": all_tasks[task_id],
                "test_solutions": all_solutions.get(task_id, []),
                "attempts": 0,
                "tokens": {
                    "input": rec_result["input_tokens"],
                    "output": rec_result["output_tokens"],
                    "thinking": rec_result["thinking_tokens"],
                },
            }

    results = {}
    total = len(pending)
    lock = threading.Lock()

    print(f"\n=== TRANSFORM PHASE ({total} tasks, {max_concurrent} concurrent, max {max_attempts} attempts each) ===")

    attempt_round = 0
    while pending:
        attempt_round += 1
        print(f"\n--- Attempt round {attempt_round} ({len(pending)} tasks remaining) ---")

        def generate_prog(task_id):
            info = pending[task_id]
            gen_result = generate_transform(client, info["recognizer_code"], info["task_data"])

            if gen_result["code"] is None:
                return task_id, {
                    "success": False,
                    "error": "No transform code parsed",
                    "input_tokens": gen_result["input_tokens"],
                    "output_tokens": gen_result["output_tokens"],
                    "thinking_tokens": gen_result["thinking_tokens"],
                }

            test_result = test_transform_full(
                info["recognizer_code"],
                gen_result["code"],
                info["task_data"],
                info["test_solutions"],
            )
            return task_id, {
                "success": test_result["success"],
                "code": gen_result["code"] if test_result["success"] else None,
                "train_passed": test_result["train_passed"],
                "train_total": test_result["train_total"],
                "test_passed": test_result["test_passed"],
                "test_total": test_result["test_total"],
                "errors": test_result.get("errors", []),
                "input_tokens": gen_result["input_tokens"],
                "output_tokens": gen_result["output_tokens"],
                "thinking_tokens": gen_result["thinking_tokens"],
            }

        with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
            futures = {executor.submit(generate_prog, task_id): task_id for task_id in list(pending.keys())}

            for future in as_completed(futures):
                task_id, result = future.result()

                with lock:
                    info = pending[task_id]
                    info["attempts"] += 1
                    info["tokens"]["input"] += result["input_tokens"]
                    info["tokens"]["output"] += result["output_tokens"]
                    info["tokens"]["thinking"] += result["thinking_tokens"]

                    if result["success"]:
                        # Build final result
                        cost = (info["tokens"]["input"] * INPUT_PRICE / 1_000_000) + \
                               (info["tokens"]["output"] * OUTPUT_PRICE / 1_000_000)
                        results[task_id] = {
                            "task_id": task_id,
                            "success": True,
                            "recognizer_attempt": 1,  # Will be updated by caller for round 2
                            "transform_attempt": info["attempts"],
                            "recognizer_code": info["recognizer_code"],
                            "transform_code": result["code"],
                            "train_passed": result["train_passed"],
                            "train_total": result["train_total"],
                            "test_passed": result["test_passed"],
                            "test_total": result["test_total"],
                            "total_input_tokens": info["tokens"]["input"],
                            "total_output_tokens": info["tokens"]["output"],
                            "total_thinking_tokens": info["tokens"]["thinking"],
                            "total_cost_usd": round(cost, 4),
                        }
                        del pending[task_id]
                        print(f"  {task_id}: ✓ SUCCESS (attempt {info['attempts']})")

                    elif info["attempts"] >= max_attempts:
                        # Max attempts reached, mark as failed
                        cost = (info["tokens"]["input"] * INPUT_PRICE / 1_000_000) + \
                               (info["tokens"]["output"] * OUTPUT_PRICE / 1_000_000)
                        results[task_id] = {
                            "task_id": task_id,
                            "success": False,
                            "transform_attempts": info["attempts"],
                            "total_input_tokens": info["tokens"]["input"],
                            "total_output_tokens": info["tokens"]["output"],
                            "total_thinking_tokens": info["tokens"]["thinking"],
                            "total_cost_usd": round(cost, 4),
                        }
                        del pending[task_id]
                        print(f"  {task_id}: ✗ FAILED after {max_attempts} attempts")

    success_count = sum(1 for r in results.values() if r["success"])
    print(f"\nTransform phase complete: {success_count}/{total} successful")
    return results


def run_validation_batched(
    task_ids: list,
    all_tasks: dict,
    all_solutions: dict,
    max_concurrent: int = 64,
    max_recognizer_rounds: int = 2,
    max_transform_attempts: int = 4,
):
    """Run batched Prolog validation with separate recognizer and transform phases."""
    completed = load_completed_tasks()
    remaining = [t for t in task_ids if t not in completed]

    print(f"\nTotal tasks: {len(task_ids)}")
    print(f"Already completed: {len(completed)}")
    print(f"Remaining: {len(remaining)}")

    if not remaining:
        print("All tasks already validated!")
        return

    client = get_client()
    all_results = {}
    remaining_set = set(remaining)
    start_time = time.time()

    for round_num in range(1, max_recognizer_rounds + 1):
        if not remaining_set:
            break

        print(f"\n{'='*60}")
        print(f"ROUND {round_num}/{max_recognizer_rounds} ({len(remaining_set)} tasks)")
        print(f"{'='*60}")

        # Phase A: Generate recognizers
        recognizers = run_recognizer_phase(
            list(remaining_set), all_tasks, all_solutions, client, max_concurrent
        )

        # Phase B: Generate and test transforms
        round_results = run_transform_phase(
            recognizers, all_tasks, all_solutions, client, max_concurrent, max_transform_attempts
        )

        # Update recognizer_attempt for this round
        for task_id, result in round_results.items():
            result["recognizer_attempt"] = round_num
            all_results[task_id] = result
            save_result(result)

            if result["success"]:
                remaining_set.discard(task_id)

        # Handle tasks that failed recognizer phase (no transform attempted)
        for task_id in list(remaining_set):
            if task_id in recognizers and not recognizers[task_id]["success"]:
                if round_num == max_recognizer_rounds:
                    # Final round, mark as failed
                    rec = recognizers[task_id]
                    cost = (rec["input_tokens"] * INPUT_PRICE / 1_000_000) + \
                           (rec["output_tokens"] * OUTPUT_PRICE / 1_000_000)
                    result = {
                        "task_id": task_id,
                        "success": False,
                        "error": "Recognizer failed",
                        "recognizer_attempts": round_num,
                        "total_input_tokens": rec["input_tokens"],
                        "total_output_tokens": rec["output_tokens"],
                        "total_thinking_tokens": rec["thinking_tokens"],
                        "total_cost_usd": round(cost, 4),
                    }
                    all_results[task_id] = result
                    save_result(result)
                    remaining_set.discard(task_id)

        # Print round summary
        round_success = sum(1 for r in round_results.values() if r["success"])
        print(f"\nRound {round_num} complete: {round_success}/{len(round_results)} successful")

    # Final summary
    elapsed = time.time() - start_time
    total_success = sum(1 for r in all_results.values() if r["success"])
    total_cost = sum(r.get("total_cost_usd", 0) for r in all_results.values())

    stats = {
        "total": len(task_ids),
        "completed": len(all_results),
        "success": total_success,
        "failed": len(all_results) - total_success,
        "total_cost_usd": round(total_cost, 4),
    }
    save_summary(stats)

    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"Tasks completed: {len(all_results)}/{len(remaining)}")
    print(f"Success: {total_success} ({100*total_success/max(1,len(all_results)):.1f}%)")
    print(f"Failed: {len(all_results) - total_success}")
    print(f"Total cost: ${total_cost:.4f}")
    print(f"Time elapsed: {elapsed/60:.1f} minutes")
    print(f"Avg time/task: {elapsed/max(1,len(all_results)):.1f}s")


def main():
    parser = argparse.ArgumentParser(description="Validate all ARC tasks with Prolog (batched)")
    parser.add_argument("--sample", type=int, help="Run on N random tasks only")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--max-concurrent", type=int, default=64, help="Max concurrent API calls")
    parser.add_argument("--max-recognizer-rounds", type=int, default=2, help="Max recognizer attempts per task")
    parser.add_argument("--max-transform-attempts", type=int, default=4, help="Max transform attempts per recognizer")
    parser.add_argument("--clear", action="store_true", help="Clear previous results and start fresh")
    args = parser.parse_args()

    # Load all tasks and solutions
    with open(TASKS_FILE) as f:
        all_tasks = json.load(f)
    with open(SOLUTIONS_FILE) as f:
        all_solutions = json.load(f)

    task_ids = sorted(all_tasks.keys())
    print(f"Loaded {len(task_ids)} tasks")

    if args.clear:
        if RESULTS_FILE.exists():
            RESULTS_FILE.unlink()
        if SUMMARY_FILE.exists():
            SUMMARY_FILE.unlink()
        print("Cleared previous results")

    if args.sample:
        random.seed(42)
        task_ids = random.sample(task_ids, min(args.sample, len(task_ids)))
        print(f"Sampled {len(task_ids)} tasks")

    run_validation_batched(
        task_ids,
        all_tasks,
        all_solutions,
        max_concurrent=args.max_concurrent,
        max_recognizer_rounds=args.max_recognizer_rounds,
        max_transform_attempts=args.max_transform_attempts,
    )


if __name__ == "__main__":
    main()
