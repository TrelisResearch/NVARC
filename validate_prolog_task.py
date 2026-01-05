"""
Prolog-based ARC task validation using pyswip.

Run: uv run validate_prolog_task.py --task 00576224
"""
import argparse
import json
import os
import random
import re
import signal
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI
from pyswip import Prolog

# Global lock for pyswip - pyswip uses a single global Prolog process
# and is not thread-safe. All Prolog operations must be serialized.
_prolog_lock = threading.Lock()

# Timeout for Prolog queries (seconds)
PROLOG_TIMEOUT = 5


class PrologTimeoutError(Exception):
    """Raised when a Prolog query exceeds the timeout."""
    pass


def _timeout_handler(signum, frame):
    """Signal handler for Prolog query timeout."""
    raise PrologTimeoutError("Prolog query timed out")


def query_with_timeout(prolog: Prolog, query: str, timeout_sec: int = PROLOG_TIMEOUT) -> list:
    """Run a Prolog query with a timeout.

    Returns list of results (empty if query fails).
    Raises PrologTimeoutError if query exceeds timeout.
    """
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout_sec)
    try:
        results = list(prolog.query(query, maxresult=1))
        return results
    finally:
        signal.alarm(0)  # Cancel the alarm

from llm_utils import (
    PROMPTS_DIR,
    TASKS_FILE,
    SOLUTIONS_FILE,
    INPUT_PRICE,
    OUTPUT_PRICE,
    get_client,
    call_gemini,
)


def grid_to_prolog(grid: list) -> str:
    """Convert a grid (list of lists) to Prolog list format.

    Example: [[1,2],[3,4]] -> "[[1,2],[3,4]]"
    """
    return str(grid).replace(" ", "")


def parse_prolog_code(response: str) -> str | None:
    """Extract Prolog code from LLM response.

    Looks for ```prolog ... ``` blocks.
    """
    codes = re.findall(r"```prolog(.*?)```", response, re.DOTALL)
    if not codes:
        # Try without language specifier
        codes = re.findall(r"```(.*?)```", response, re.DOTALL)
    if not codes:
        return None
    longest_code = max(codes, key=len)
    return longest_code.strip()


def create_prolog_engine(code: str) -> Prolog:
    """Create Prolog engine by consulting code from temp file.

    Note: pyswip shares global Prolog state, so predicates may be redefined.
    Warnings about redefined predicates are expected and harmless.
    """
    prolog = Prolog()

    with tempfile.NamedTemporaryFile(mode='w', suffix='.pl', delete=False) as f:
        f.write(code)
        temp_path = f.name

    try:
        prolog.consult(temp_path)
    finally:
        os.unlink(temp_path)

    return prolog


def test_valid_input(prolog: Prolog, grid: list) -> bool:
    """Test if valid_input(Grid) succeeds for the given grid.

    Returns True if the predicate succeeds, False otherwise.
    Raises PrologTimeoutError if query times out.
    """
    grid_str = grid_to_prolog(grid)
    query = f"valid_input({grid_str})"

    try:
        results = query_with_timeout(prolog, query)
        return len(results) > 0
    except PrologTimeoutError:
        raise  # Re-raise timeout so caller can handle it
    except Exception:
        return False


def test_transform(prolog: Prolog, input_grid: list, expected_output: list) -> bool:
    """Test if transform(Input, Output) produces the expected output.

    Returns True if Output unifies with expected_output.
    Raises PrologTimeoutError if query times out.
    """
    input_str = grid_to_prolog(input_grid)
    expected_str = grid_to_prolog(expected_output)

    # Query: transform(Input, Output), Output = Expected
    query = f"transform({input_str}, Output), Output = {expected_str}"

    try:
        results = query_with_timeout(prolog, query)
        return len(results) > 0
    except PrologTimeoutError:
        raise  # Re-raise timeout so caller can handle it
    except Exception:
        return False


def test_specificity(prolog: Prolog, train_grids: list, rejection_threshold: float = 0.9) -> dict:
    """Ensure recognizer rejects random grids (not underconstrained).

    A good recognizer should reject most random grids since valid ARC inputs
    have specific structure. Default 90% = must reject 9/10 random grids.

    Returns dict with rejections count and success status.
    """
    # Get dimension bounds from train grids
    min_rows = min(len(g) for g in train_grids)
    max_rows = max(len(g) for g in train_grids)
    min_cols = min(len(g[0]) for g in train_grids if g)
    max_cols = max(len(g[0]) for g in train_grids if g)

    rejections = 0
    total_tests = 10

    for _ in range(total_tests):
        # Generate random grid with similar dimensions
        rows = random.randint(max(1, min_rows - 1), max_rows + 1)
        cols = random.randint(max(1, min_cols - 1), max_cols + 1)
        random_grid = [[random.randint(0, 9) for _ in range(cols)] for _ in range(rows)]

        try:
            if not test_valid_input(prolog, random_grid):
                rejections += 1
        except PrologTimeoutError:
            # Timeout counts as rejection (couldn't validate)
            rejections += 1

    required_rejections = int(total_tests * rejection_threshold)
    return {
        "rejections": rejections,
        "total_tests": total_tests,
        "threshold": required_rejections,
        "success": rejections >= required_rejections,
    }


def format_input_recognizer_prompt(task_data: dict) -> str:
    """Format prompt for input recognizer generation.

    IMPORTANT: Only uses train INPUT grids, not outputs or test inputs.
    """
    template = (PROMPTS_DIR / "prolog_input_recognizer.md").read_text()

    # Format train inputs only
    inputs_text = []
    for i, ex in enumerate(task_data["train"]):
        input_grid = ex["input"]
        rows = len(input_grid)
        cols = len(input_grid[0]) if input_grid else 0
        inputs_text.append(f"Example {i+1} ({rows}x{cols}):\n{grid_to_prolog(input_grid)}")

    prompt = template.replace("{TRAIN_INPUTS}", "\n\n".join(inputs_text))
    return prompt


def format_transform_prompt(recognizer_code: str, task_data: dict) -> str:
    """Format prompt for transform/2 generation.

    Provides: recognizer code + train input/output pairs.
    """
    template = (PROMPTS_DIR / "prolog_transform.md").read_text()

    # Format train pairs
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
    prompt = prompt.replace("{TRAIN_PAIRS}", "\n".join(pairs_text))
    return prompt


def generate_input_recognizer(client: OpenAI, task_data: dict) -> dict:
    """Generate valid_input/1 predicate from train inputs.

    Returns dict with code and token usage.
    """
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

    Returns results dict with pass/fail counts.
    Thread-safe: uses global lock for pyswip operations.
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

    # Lock all Prolog operations - pyswip is not thread-safe
    with _prolog_lock:
        try:
            prolog = create_prolog_engine(code)
        except Exception as e:
            results["errors"].append(f"Prolog syntax error: {str(e)[:100]}")
            return results

        # Test train inputs
        for i, grid in enumerate(train_inputs):
            try:
                if test_valid_input(prolog, grid):
                    results["train_passed"] += 1
                else:
                    results["errors"].append(f"Train {i}: valid_input failed")
            except Exception as e:
                results["errors"].append(f"Train {i}: {type(e).__name__}: {str(e)[:50]}")

        # Test test inputs (without leaking to LLM)
        for i, grid in enumerate(test_inputs):
            try:
                if test_valid_input(prolog, grid):
                    results["test_passed"] += 1
                else:
                    results["errors"].append(f"Test {i}: valid_input failed")
            except Exception as e:
                results["errors"].append(f"Test {i}: {type(e).__name__}: {str(e)[:50]}")

        # Specificity test - must reject random grids
        specificity = test_specificity(prolog, train_inputs)
        results["specificity_passed"] = specificity["success"]
        results["specificity_rejections"] = specificity["rejections"]
        if not specificity["success"]:
            results["errors"].append(
                f"Specificity failed: rejected {specificity['rejections']}/{specificity['total_tests']} "
                f"(need >= {specificity['threshold']})"
            )

    # Success = all train + all test + specificity
    results["success"] = (
        results["train_passed"] == results["train_total"] and
        results["test_passed"] == results["test_total"] and
        results["specificity_passed"]
    )
    return results


def generate_transform(client: OpenAI, recognizer_code: str, task_data: dict) -> dict:
    """Generate transform/2 predicate.

    Returns dict with code and token usage.
    """
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

    Exact match required for success.
    Thread-safe: uses global lock for pyswip operations.
    """
    results = {
        "train_passed": 0,
        "train_total": len(task_data["train"]),
        "test_passed": 0,
        "test_total": len(task_data.get("test", [])),
        "errors": [],
    }

    # Combine recognizer + transform code
    full_code = recognizer_code + "\n\n" + transform_code

    # Lock all Prolog operations - pyswip is not thread-safe
    with _prolog_lock:
        try:
            prolog = create_prolog_engine(full_code)
        except Exception as e:
            results["errors"].append(f"Prolog syntax error: {str(e)[:100]}")
            return results

        # Test train pairs
        for i, ex in enumerate(task_data["train"]):
            try:
                if test_transform(prolog, ex["input"], ex["output"]):
                    results["train_passed"] += 1
                else:
                    results["errors"].append(f"Train {i}: transform mismatch")
            except Exception as e:
                results["errors"].append(f"Train {i}: {type(e).__name__}: {str(e)[:50]}")

        # Test test pairs
        for i, ex in enumerate(task_data.get("test", [])):
            if i >= len(test_solutions):
                continue
            try:
                if test_transform(prolog, ex["input"], test_solutions[i]):
                    results["test_passed"] += 1
                else:
                    results["errors"].append(f"Test {i}: transform mismatch")
            except Exception as e:
                results["errors"].append(f"Test {i}: {type(e).__name__}: {str(e)[:50]}")

    results["success"] = (
        results["train_passed"] == results["train_total"] and
        results["test_passed"] == results["test_total"]
    )
    return results


def generate_and_test_transform(
    client: OpenAI,
    recognizer_code: str,
    task_data: dict,
    test_solutions: list,
    attempt: int,
) -> dict:
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
    """Main validation loop for a single task using Prolog.

    Two-phase approach:
    1. Generate input recognizer (multiple attempts)
    2. For each successful recognizer, generate transforms (multiple attempts)
    """
    # Load task
    with open(TASKS_FILE) as f:
        all_tasks = json.load(f)
    if task_id not in all_tasks:
        return {"task_id": task_id, "success": False, "error": f"Task {task_id} not found"}
    task_data = all_tasks[task_id]

    # Load solutions
    with open(SOLUTIONS_FILE) as f:
        all_solutions = json.load(f)
    test_solutions = all_solutions.get(task_id, [])

    # Extract test inputs (for recognizer validation only, not shown to LLM)
    test_inputs = [ex["input"] for ex in task_data.get("test", [])]

    client = get_client()
    total_input = total_output = total_thinking = 0

    for rec_attempt in range(1, recognizer_attempts + 1):
        print(f"\n=== Recognizer attempt {rec_attempt}/{recognizer_attempts} ===")

        # Phase 1: Generate input recognizer
        rec_result = generate_input_recognizer(client, task_data)
        total_input += rec_result["input_tokens"]
        total_output += rec_result["output_tokens"]
        total_thinking += rec_result["thinking_tokens"]

        if rec_result["code"] is None:
            print(f"Failed to parse recognizer code (attempt {rec_attempt})")
            continue

        print(f"Recognizer generated ({len(rec_result['code'])} chars)")

        # Test recognizer
        rec_test = test_input_recognizer(rec_result["code"], task_data, test_inputs)

        train_status = f"{rec_test['train_passed']}/{rec_test['train_total']}"
        test_status = f"{rec_test['test_passed']}/{rec_test['test_total']}"
        spec_status = "PASS" if rec_test["specificity_passed"] else "FAIL"

        print(f"  Train: {train_status}, Test: {test_status}, Specificity: {spec_status}")

        if not rec_test["success"]:
            print(f"  Errors: {rec_test['errors'][:2]}")
            continue

        print(f"Recognizer passed! Testing {transform_attempts} transforms...")

        # Phase 2: Generate transforms (concurrent)
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
                    # Cancel remaining and return success
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

    # All attempts failed
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
    # Print without the code for brevity
    result_summary = {k: v for k, v in result.items() if not k.endswith("_code")}
    print(json.dumps(result_summary, indent=2))

    if result["success"]:
        print("\n--- Recognizer Code ---")
        print(result["recognizer_code"][:500] + "..." if len(result.get("recognizer_code", "")) > 500 else result.get("recognizer_code", ""))
        print("\n--- Transform Code ---")
        print(result["transform_code"][:500] + "..." if len(result.get("transform_code", "")) > 500 else result.get("transform_code", ""))

    # Save result
    output_file = f"prolog_validation_result_{args.task}.json"
    with open(output_file, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to: {output_file}")


if __name__ == "__main__":
    main()
