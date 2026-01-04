"""
Validate all ARC-AGI-2 training tasks with parallel execution.

Run: uv run validate_all_tasks.py
     uv run validate_all_tasks.py --sample 10              # test on 10 tasks first
     uv run validate_all_tasks.py --parallel-tasks 16      # run 16 tasks in parallel
     uv run validate_all_tasks.py --resume                 # resume from checkpoint
"""
import argparse
import json
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from validate_task import validate_task, TASKS_FILE

RESULTS_FILE = Path(__file__).parent / "validation_results.jsonl"
SUMMARY_FILE = Path(__file__).parent / "validation_summary.json"

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


def run_validation(task_ids: list, desc_attempts: int, prog_attempts: int, concurrent: int, parallel_tasks: int = 32):
    """Run validation on a list of tasks with progress tracking and parallel execution."""
    completed = load_completed_tasks()
    remaining = [t for t in task_ids if t not in completed]

    print(f"\nTotal tasks: {len(task_ids)}")
    print(f"Already completed: {len(completed)}")
    print(f"Remaining: {len(remaining)}")

    if not remaining:
        print("All tasks already validated!")
        return

    stats = {
        "total": len(task_ids),
        "completed": len(completed),
        "success": 0,
        "failed": 0,
        "total_cost_usd": 0.0,
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "total_thinking_tokens": 0,
    }

    # Load existing stats
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    if r.get("success"):
                        stats["success"] += 1
                    else:
                        stats["failed"] += 1
                    stats["total_cost_usd"] += r.get("total_cost_usd", 0)
                    stats["total_input_tokens"] += r.get("total_input_tokens", 0)
                    stats["total_output_tokens"] += r.get("total_output_tokens", 0)
                    stats["total_thinking_tokens"] += r.get("total_thinking_tokens", 0)

    start_time = time.time()
    stats_lock = threading.Lock()
    completed_count = [0]  # Mutable container for tracking progress

    def process_task(task_id):
        """Process a single task and return the result."""
        try:
            return validate_task(
                task_id,
                description_attempts=desc_attempts,
                output_program_attempts=prog_attempts,
                concurrent_requests=concurrent,
            )
        except Exception as e:
            return {"task_id": task_id, "success": False, "error": str(e)}

    print(f"\nRunning {len(remaining)} tasks with {parallel_tasks} parallel workers...")

    try:
        with ThreadPoolExecutor(max_workers=parallel_tasks) as executor:
            futures = {executor.submit(process_task, task_id): task_id for task_id in remaining}

            for future in as_completed(futures):
                task_id = futures[future]
                result = future.result()

                save_result(result)

                with stats_lock:
                    stats["completed"] += 1
                    completed_count[0] += 1

                    if result.get("success"):
                        stats["success"] += 1
                        status = f"✓ SUCCESS (desc={result.get('description_attempt')}, prog={result.get('program_attempt')})"
                    else:
                        stats["failed"] += 1
                        error_msg = result.get("error", "")
                        status = f"✗ FAILED" + (f": {error_msg}" if error_msg else "")

                    stats["total_cost_usd"] += result.get("total_cost_usd", 0)
                    stats["total_input_tokens"] += result.get("total_input_tokens", 0)
                    stats["total_output_tokens"] += result.get("total_output_tokens", 0)
                    stats["total_thinking_tokens"] += result.get("total_thinking_tokens", 0)

                    print(f"[{completed_count[0]}/{len(remaining)}] {task_id}: {status} | Total: ${stats['total_cost_usd']:.4f}")
                    save_summary(stats)

    except KeyboardInterrupt:
        print("\n\nInterrupted! Progress saved.")
        save_summary(stats)

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"Tasks completed: {stats['completed']}/{stats['total']}")
    print(f"Success: {stats['success']} ({100*stats['success']/max(1,stats['completed']):.1f}%)")
    print(f"Failed: {stats['failed']}")
    print(f"Total cost: ${stats['total_cost_usd']:.4f}")
    print(f"Time elapsed: {elapsed/60:.1f} minutes")
    print(f"Avg cost/task: ${stats['total_cost_usd']/max(1,stats['completed']):.4f}")
    print(f"Avg time/task: {elapsed/max(1,len(remaining)-len(task_ids)+stats['completed']):.1f}s")


def main():
    parser = argparse.ArgumentParser(description="Validate all ARC tasks")
    parser.add_argument("--sample", type=int, help="Run on N random tasks only")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--description-attempts", type=int, default=2)
    parser.add_argument("--output-program-attempts", type=int, default=8)
    parser.add_argument("--concurrent-requests", type=int, default=4)
    parser.add_argument("--parallel-tasks", type=int, default=32, help="Number of tasks to run in parallel")
    parser.add_argument("--clear", action="store_true", help="Clear previous results and start fresh")
    args = parser.parse_args()

    # Load all task IDs
    with open(TASKS_FILE) as f:
        all_tasks = json.load(f)
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

    run_validation(
        task_ids,
        desc_attempts=args.description_attempts,
        prog_attempts=args.output_program_attempts,
        concurrent=args.concurrent_requests,
        parallel_tasks=args.parallel_tasks,
    )


if __name__ == "__main__":
    main()
