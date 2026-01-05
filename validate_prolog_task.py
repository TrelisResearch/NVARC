"""
Prolog-based ARC task validation using subprocess-isolated Prolog.

Run: uv run validate_prolog_task.py --task 00576224
"""
import argparse
import json
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

from llm_utils import (
    PROMPTS_DIR,
    TASKS_FILE,
    SOLUTIONS_FILE,
    INPUT_PRICE,
    OUTPUT_PRICE,
    get_client,
    call_gemini,
)
from prolog_utils import (
    PrologTimeoutError,
    grid_to_prolog,
    parse_prolog_code,
    test_valid_input,
    test_transform,
)


def test_specificity(code: str, train_grids: list, rejection_threshold: float = 0.9) -> dict:
    """Ensure recognizer rejects random grids (not underconstrained).

    A good recognizer should reject most random grids since valid ARC inputs
    have specific structure. Default 90% = must reject 9/10 random grids.

    Returns dict with rejections count and success status.
    """
    min_rows = min(len(g) for g in train_grids)
    max_rows = max(len(g) for g in train_grids)
    min_cols = min(len(g[0]) for g in train_grids if g)
    max_cols = max(len(g[0]) for g in train_grids if g)

    rejections = 0
    total_tests = 10

    for _ in range(total_tests):
        rows = random.randint(max(1, min_rows - 1), max_rows + 1)
        cols = random.randint(max(1, min_cols - 1), max_cols + 1)
        random_grid = [[random.randint(0, 9) for _ in range(cols)] for _ in range(rows)]

        try:
            if not test_valid_input(code, random_grid):
                rejections += 1
        except PrologTimeoutError:
            rejections += 1

    required_rejections = int(total_tests * rejection_threshold)
    return {
        "rejections": rejections,
        "total_tests": total_tests,
        "threshold": required_rejections,
        "success": rejections >= required_rejections,
    }


def format_input_recognizer_prompt(task_data: dict) -> str:
    """Format prompt for input recognizer generation."""
    template = (PROMPTS_DIR / "prolog_input_recognizer.md").read_text()

    inputs_text = []
    for i, ex in enumerate(task_data["train"]):
        input_grid = ex["input"]
        rows = len(input_grid)
        cols = len(input_grid[0]) if input_grid else 0
        inputs_text.append(f"Example {i+1} ({rows}x{cols}):\n{grid_to_prolog(input_grid)}")

    return template.replace("{TRAIN_INPUTS}", "\n\n".join(inputs_text))


def format_transform_prompt(recognizer_code: str, task_data: dict) -> str:
    """Format prompt for transform/2 generation."""
    template = (PROMPTS_DIR / "prolog_transform.md").read_text()

    pairs_text = []
    for i, ex in enumerate(task_data["train"]):
        input_grid = ex["input"]
        output_grid = ex["output"]
        in_rows, in_cols = len(input_grid), len(input_grid[0]) if input_grid else 0
        out_rows, out_cols = len(output_grid), len(output_grid[0]) if output_grid else 0
        pairs_text.append(f"Example {i+1}:")
        pairs_text.append(f"Input ({in_rows}x{in_cols}): {grid_to_prolog(input_grid)}")
        pairs_text.append(f"Output ({out_rows}x{out_cols}): {grid_to_prolog(output_grid)}")
        pairs_text.append("")

    prompt = template.replace("{RECOGNIZER_CODE}", recognizer_code)
    return prompt.replace("{TRAIN_PAIRS}", "\n".join(pairs_text))


def generate_input_recognizer(client, task_data: dict) -> dict:
    """Generate valid_input/1 predicate from train inputs."""
    prompt = format_input_recognizer_prompt(task_data)
    response = call_gemini(client, prompt)

    code = parse_prolog_code(response["content"])
    return {
        "code": code,
        "raw_response": response["content"],
        "input_tokens": response["input_tokens"],
        "output_tokens": response["output_tokens"],
        "thinking_tokens": response["thinking_tokens"],
    }


def test_input_recognizer(code: str, task_data: dict, test_inputs: list) -> dict:
    """Test input recognizer on train AND test inputs, plus specificity.

    Uses subprocess isolation - no locks needed.
    """
    train_inputs = [ex["input"] for ex in task_data["train"]]

    results = {
        "train_passed": 0,
        "train_total": len(train_inputs),
        "test_passed": 0,
        "test_total": len(test_inputs),
        "specificity_passed": False,
        "errors": [],
    }

    # Test train inputs
    for i, grid in enumerate(train_inputs):
        try:
            if test_valid_input(code, grid):
                results["train_passed"] += 1
            else:
                results["errors"].append(f"Train {i}: valid_input failed")
        except PrologTimeoutError:
            results["errors"].append(f"Train {i}: timeout")
        except Exception as e:
            results["errors"].append(f"Train {i}: {type(e).__name__}: {str(e)[:50]}")

    # Test test inputs
    for i, grid in enumerate(test_inputs):
        try:
            if test_valid_input(code, grid):
                results["test_passed"] += 1
            else:
                results["errors"].append(f"Test {i}: valid_input failed")
        except PrologTimeoutError:
            results["errors"].append(f"Test {i}: timeout")
        except Exception as e:
            results["errors"].append(f"Test {i}: {type(e).__name__}: {str(e)[:50]}")

    # Specificity test
    specificity = test_specificity(code, train_inputs)
    results["specificity_passed"] = specificity["success"]
    results["specificity_rejections"] = specificity["rejections"]
    if not specificity["success"]:
        results["errors"].append(
            f"Specificity failed: rejected {specificity['rejections']}/{specificity['total_tests']} "
            f"(need >= {specificity['threshold']})"
        )

    results["success"] = (
        results["train_passed"] == results["train_total"] and
        results["test_passed"] == results["test_total"] and
        results["specificity_passed"]
    )
    return results


def generate_transform(client, recognizer_code: str, task_data: dict) -> dict:
    """Generate transform/2 predicate."""
    prompt = format_transform_prompt(recognizer_code, task_data)
    response = call_gemini(client, prompt)

    code = parse_prolog_code(response["content"])
    return {
        "code": code,
        "raw_response": response["content"],
        "input_tokens": response["input_tokens"],
        "output_tokens": response["output_tokens"],
        "thinking_tokens": response["thinking_tokens"],
    }


def test_transform_full(recognizer_code: str, transform_code: str, task_data: dict, test_solutions: list) -> dict:
    """Test transform/2 on train AND test pairs.

    Uses subprocess isolation - no locks needed.
    """
    results = {
        "train_passed": 0,
        "train_total": len(task_data["train"]),
        "test_passed": 0,
        "test_total": len(task_data.get("test", [])),
        "errors": [],
    }

    full_code = recognizer_code + "\n\n" + transform_code

    # Test train pairs
    for i, ex in enumerate(task_data["train"]):
        try:
            if test_transform(full_code, ex["input"], ex["output"]):
                results["train_passed"] += 1
            else:
                results["errors"].append(f"Train {i}: transform mismatch")
        except PrologTimeoutError:
            results["errors"].append(f"Train {i}: timeout")
        except Exception as e:
            results["errors"].append(f"Train {i}: {type(e).__name__}: {str(e)[:50]}")

    # Test test pairs
    for i, ex in enumerate(task_data.get("test", [])):
        if i >= len(test_solutions):
            continue
        try:
            if test_transform(full_code, ex["input"], test_solutions[i]):
                results["test_passed"] += 1
            else:
                results["errors"].append(f"Test {i}: transform mismatch")
        except PrologTimeoutError:
            results["errors"].append(f"Test {i}: timeout")
        except Exception as e:
            results["errors"].append(f"Test {i}: {type(e).__name__}: {str(e)[:50]}")

    results["success"] = (
        results["train_passed"] == results["train_total"] and
        results["test_passed"] == results["test_total"]
    )
    return results


def generate_and_test_transform(client, recognizer_code: str, task_data: dict, test_solutions: list, attempt: int) -> dict:
    """Generate and test a single transform attempt."""
    gen_result = generate_transform(client, recognizer_code, task_data)

    if gen_result["code"] is None:
        return {
            "success": False,
            "attempt": attempt,
            "error": "No Prolog code found in response",
            "input_tokens": gen_result["input_tokens"],
            "output_tokens": gen_result["output_tokens"],
            "thinking_tokens": gen_result["thinking_tokens"],
        }

    test_result = test_transform_full(recognizer_code, gen_result["code"], task_data, test_solutions)
    return {
        "success": test_result["success"],
        "attempt": attempt,
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


def validate_prolog_task(
    task_id: str,
    recognizer_attempts: int = 2,
    transform_attempts: int = 4,
    concurrent_requests: int = 4,
) -> dict:
    """Main validation loop for a single task using Prolog."""
    with open(TASKS_FILE) as f:
        all_tasks = json.load(f)
    if task_id not in all_tasks:
        return {"task_id": task_id, "success": False, "error": f"Task {task_id} not found"}
    task_data = all_tasks[task_id]

    with open(SOLUTIONS_FILE) as f:
        all_solutions = json.load(f)
    test_solutions = all_solutions.get(task_id, [])

    test_inputs = [ex["input"] for ex in task_data.get("test", [])]

    client = get_client()
    total_input = total_output = total_thinking = 0

    for rec_attempt in range(1, recognizer_attempts + 1):
        print(f"\n=== Recognizer attempt {rec_attempt}/{recognizer_attempts} ===")

        rec_result = generate_input_recognizer(client, task_data)
        total_input += rec_result["input_tokens"]
        total_output += rec_result["output_tokens"]
        total_thinking += rec_result["thinking_tokens"]

        if rec_result["code"] is None:
            print(f"Failed to parse recognizer code (attempt {rec_attempt})")
            continue

        print(f"Recognizer generated ({len(rec_result['code'])} chars)")

        rec_test = test_input_recognizer(rec_result["code"], task_data, test_inputs)

        train_status = f"{rec_test['train_passed']}/{rec_test['train_total']}"
        test_status = f"{rec_test['test_passed']}/{rec_test['test_total']}"
        spec_status = "PASS" if rec_test["specificity_passed"] else "FAIL"

        print(f"  Train: {train_status}, Test: {test_status}, Specificity: {spec_status}")

        if not rec_test["success"]:
            print(f"  Errors: {rec_test['errors'][:2]}")
            continue

        print(f"Recognizer passed! Testing {transform_attempts} transforms...")

        with ThreadPoolExecutor(max_workers=concurrent_requests) as executor:
            futures = {
                executor.submit(
                    generate_and_test_transform,
                    client,
                    rec_result["code"],
                    task_data,
                    test_solutions,
                    i + 1,
                ): i
                for i in range(transform_attempts)
            }

            for future in as_completed(futures):
                result = future.result()
                total_input += result["input_tokens"]
                total_output += result["output_tokens"]
                total_thinking += result["thinking_tokens"]

                train_status = f"{result.get('train_passed', 0)}/{result.get('train_total', '?')}"
                test_status = f"{result.get('test_passed', 0)}/{result.get('test_total', '?')}"
                status = "PASS" if result["success"] else f"FAIL (train={train_status}, test={test_status})"
                print(f"  Transform {result['attempt']}: {status}")

                if result["success"]:
                    for f in futures:
                        f.cancel()

                    cost = (total_input * INPUT_PRICE / 1_000_000) + \
                           (total_output * OUTPUT_PRICE / 1_000_000)

                    return {
                        "task_id": task_id,
                        "success": True,
                        "recognizer_attempt": rec_attempt,
                        "transform_attempt": result["attempt"],
                        "recognizer_code": rec_result["code"],
                        "transform_code": result["code"],
                        "train_passed": result["train_passed"],
                        "train_total": result["train_total"],
                        "test_passed": result["test_passed"],
                        "test_total": result["test_total"],
                        "total_input_tokens": total_input,
                        "total_output_tokens": total_output,
                        "total_thinking_tokens": total_thinking,
                        "total_cost_usd": round(cost, 4),
                    }

    cost = (total_input * INPUT_PRICE / 1_000_000) + (total_output * OUTPUT_PRICE / 1_000_000)
    return {
        "task_id": task_id,
        "success": False,
        "recognizer_attempts_tried": recognizer_attempts,
        "transform_attempts_per_rec": transform_attempts,
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "total_thinking_tokens": total_thinking,
        "total_cost_usd": round(cost, 4),
    }


def main():
    parser = argparse.ArgumentParser(description="Validate ARC task with Prolog")
    parser.add_argument("--task", required=True, help="Task ID (e.g., 00576224)")
    parser.add_argument("--recognizer-attempts", type=int, default=2,
                        help="Number of recognizer attempts (default: 2)")
    parser.add_argument("--transform-attempts", type=int, default=4,
                        help="Transform attempts per recognizer (default: 4)")
    parser.add_argument("--concurrent-requests", type=int, default=4,
                        help="Concurrent API calls (default: 4)")
    args = parser.parse_args()

    print(f"Validating task: {args.task}")
    print(f"Config: {args.recognizer_attempts} recognizer attempts, "
          f"{args.transform_attempts} transforms each, {args.concurrent_requests} concurrent")

    result = validate_prolog_task(
        args.task,
        recognizer_attempts=args.recognizer_attempts,
        transform_attempts=args.transform_attempts,
        concurrent_requests=args.concurrent_requests,
    )

    print("\n" + "=" * 60)
    print("RESULT:")
    result_summary = {k: v for k, v in result.items() if not k.endswith("_code")}
    print(json.dumps(result_summary, indent=2))

    if result["success"]:
        print("\n--- Recognizer Code ---")
        print(result["recognizer_code"][:500] + "..." if len(result.get("recognizer_code", "")) > 500 else result.get("recognizer_code", ""))
        print("\n--- Transform Code ---")
        print(result["transform_code"][:500] + "..." if len(result.get("transform_code", "")) > 500 else result.get("transform_code", ""))

    output_file = f"prolog_validation_result_{args.task}.json"
    with open(output_file, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to: {output_file}")


if __name__ == "__main__":
    main()
