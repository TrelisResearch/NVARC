"""
Single-task Prolog validation with style selection.

Run: uv run validate_prolog_single.py --task aa18de87 --style checker
     uv run validate_prolog_single.py --task aa18de87 --style generator
"""
import argparse
import json
import random
import signal
import tempfile
import os

from pyswip import Prolog

from llm_utils import (
    PROMPTS_DIR,
    TASKS_FILE,
    SOLUTIONS_FILE,
    INPUT_PRICE,
    OUTPUT_PRICE,
    get_client,
    call_gemini,
)

# Timeout for Prolog queries (seconds)
PROLOG_TIMEOUT = 5


class PrologTimeoutError(Exception):
    pass


def _timeout_handler(signum, frame):
    raise PrologTimeoutError("Prolog query timed out")


def query_with_timeout(prolog: Prolog, query: str, timeout_sec: int = PROLOG_TIMEOUT) -> list:
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout_sec)
    try:
        results = list(prolog.query(query, maxresult=1))
        return results
    finally:
        signal.alarm(0)


def grid_to_prolog(grid: list) -> str:
    return str(grid).replace(" ", "")


def parse_prolog_code(response: str) -> str | None:
    import re
    codes = re.findall(r"```prolog(.*?)```", response, re.DOTALL)
    if not codes:
        codes = re.findall(r"```(.*?)```", response, re.DOTALL)
    if not codes:
        return None
    return max(codes, key=len).strip()


def create_prolog_engine(code: str) -> Prolog:
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
    grid_str = grid_to_prolog(grid)
    query = f"valid_input({grid_str})"
    try:
        results = query_with_timeout(prolog, query)
        return len(results) > 0
    except PrologTimeoutError:
        raise
    except Exception:
        return False


def format_prompt(task_data: dict, style: str) -> str:
    """Format prompt based on style (checker or generator)."""
    if style == "checker":
        template_file = PROMPTS_DIR / "prolog_input_recognizer_checker.md"
    else:
        template_file = PROMPTS_DIR / "prolog_input_recognizer_generator.md"

    template = template_file.read_text()

    inputs_text = []
    for i, ex in enumerate(task_data["train"]):
        input_grid = ex["input"]
        rows = len(input_grid)
        cols = len(input_grid[0]) if input_grid else 0
        inputs_text.append(f"Example {i+1} ({rows}x{cols}):\n{grid_to_prolog(input_grid)}")

    return template.replace("{TRAIN_INPUTS}", "\n\n".join(inputs_text))


def test_specificity(prolog: Prolog, train_grids: list, num_tests: int = 10) -> dict:
    """Test that recognizer rejects random grids."""
    min_rows = min(len(g) for g in train_grids)
    max_rows = max(len(g) for g in train_grids)
    min_cols = min(len(g[0]) for g in train_grids if g)
    max_cols = max(len(g[0]) for g in train_grids if g)

    rejections = 0
    for _ in range(num_tests):
        rows = random.randint(max(1, min_rows - 1), max_rows + 1)
        cols = random.randint(max(1, min_cols - 1), max_cols + 1)
        random_grid = [[random.randint(0, 9) for _ in range(cols)] for _ in range(rows)]

        try:
            if not test_valid_input(prolog, random_grid):
                rejections += 1
        except PrologTimeoutError:
            rejections += 1

    return {"rejections": rejections, "total": num_tests}


def main():
    parser = argparse.ArgumentParser(description="Single-task Prolog validation")
    parser.add_argument("--task", required=True, help="Task ID")
    parser.add_argument("--style", choices=["checker", "generator"], default="checker",
                        help="Prolog style: checker (validation) or generator (CLP(FD))")
    parser.add_argument("--attempts", type=int, default=3, help="Number of attempts")
    args = parser.parse_args()

    # Load task
    with open(TASKS_FILE) as f:
        all_tasks = json.load(f)
    if args.task not in all_tasks:
        print(f"Task {args.task} not found")
        return
    task_data = all_tasks[args.task]

    train_inputs = [ex["input"] for ex in task_data["train"]]
    test_inputs = [ex["input"] for ex in task_data.get("test", [])]

    print(f"Task: {args.task}")
    print(f"Style: {args.style}")
    print(f"Train examples: {len(train_inputs)}")
    print(f"Test examples: {len(test_inputs)}")
    print()

    client = get_client()
    prompt = format_prompt(task_data, args.style)

    total_cost = 0

    for attempt in range(1, args.attempts + 1):
        print(f"=== Attempt {attempt}/{args.attempts} ===")

        # Generate recognizer
        response = call_gemini(client, prompt)
        cost = (response["input_tokens"] * INPUT_PRICE + response["output_tokens"] * OUTPUT_PRICE) / 1_000_000
        total_cost += cost

        code = parse_prolog_code(response["content"])
        if code is None:
            print("  Failed to parse Prolog code")
            continue

        print(f"  Generated {len(code)} chars of Prolog")

        # Save code for inspection
        debug_file = f"/tmp/debug_{args.style}_{args.task}_{attempt}.pl"
        with open(debug_file, "w") as f:
            f.write(code)
        print(f"  Saved to: {debug_file}")

        # Test recognizer
        try:
            prolog = create_prolog_engine(code)
        except Exception as e:
            print(f"  Prolog syntax error: {str(e)[:80]}")
            continue

        # Test train inputs
        train_passed = 0
        for i, grid in enumerate(train_inputs):
            try:
                if test_valid_input(prolog, grid):
                    train_passed += 1
                else:
                    print(f"  Train {i}: REJECTED")
            except PrologTimeoutError:
                print(f"  Train {i}: TIMEOUT")
            except Exception as e:
                print(f"  Train {i}: ERROR - {e}")

        # Test test inputs
        test_passed = 0
        for i, grid in enumerate(test_inputs):
            try:
                if test_valid_input(prolog, grid):
                    test_passed += 1
                else:
                    print(f"  Test {i}: REJECTED")
            except PrologTimeoutError:
                print(f"  Test {i}: TIMEOUT")
            except Exception as e:
                print(f"  Test {i}: ERROR - {e}")

        # Specificity test
        spec = test_specificity(prolog, train_inputs)

        print(f"  Results: train={train_passed}/{len(train_inputs)}, test={test_passed}/{len(test_inputs)}, specificity={spec['rejections']}/{spec['total']}")

        if train_passed == len(train_inputs):
            print("  SUCCESS - Recognizer accepts all train inputs!")
            if test_passed == len(test_inputs):
                print("  BONUS - Also accepts all test inputs!")
            break
        print()

    print(f"\nTotal cost: ${total_cost:.4f}")


if __name__ == "__main__":
    main()
