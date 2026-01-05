"""
Validate ARC task descriptions by generating output programs and testing on train examples.

Run: uv run validate_task.py --task 00576224
"""
import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from openai import OpenAI

from llm_utils import (
    PROMPTS_DIR,
    TASKS_FILE,
    SOLUTIONS_FILE,
    INPUT_PRICE,
    OUTPUT_PRICE,
    get_client,
    call_gemini,
)

# Add SDG/scripts to path for parser and puzzle utilities
sys.path.insert(0, str(Path(__file__).parent / "SDG" / "scripts"))
from parser import parse_python_code
from puzzle import validate_and_convert_grid
from utils import recognize_summary, summary_to_text, convert_grid_to_string


def format_task_for_prompt(task_data: dict) -> str:
    """Format task train examples for summary_v2 prompt."""
    lines = []
    for i, ex in enumerate(task_data["train"]):
        input_grid = np.array(ex["input"])
        output_grid = np.array(ex["output"])
        lines.append(f"Example {i+1}:")
        lines.append(f"Input ({input_grid.shape[0]}x{input_grid.shape[1]}):")
        lines.append(convert_grid_to_string(input_grid))
        lines.append(f"Output ({output_grid.shape[0]}x{output_grid.shape[1]}):")
        lines.append(convert_grid_to_string(output_grid))
        lines.append("")
    return "\n".join(lines)


def format_description_prompt(task_data: dict) -> str:
    """Format summary_v2.md prompt with task examples."""
    template = (PROMPTS_DIR / "summary_v2.md").read_text()
    puzzle_text = format_task_for_prompt(task_data)
    # Replace {PUZZLE} with task examples, remove {EXAMPLES} section (zero-shot)
    prompt = template.replace("{PUZZLE}", puzzle_text)
    # Remove the examples reference section since we're doing zero-shot
    prompt = prompt.replace("{EXAMPLES}", "(No reference examples provided - analyze this puzzle directly)")
    return prompt


def format_output_program_prompt(description: dict, task_data: dict) -> str:
    """Format generate_puzzle_output.md prompt with description and train examples as reference."""
    template = (PROMPTS_DIR / "generate_puzzle_output.md").read_text()
    puzzle_text = summary_to_text(description)
    # Use train examples as reference "input code" (showing what grids look like)
    input_code = "# Reference train examples (input/output pairs):\n"
    for i, ex in enumerate(task_data["train"]):
        input_grid = np.array(ex["input"])
        output_grid = np.array(ex["output"])
        input_code += f"\n# Example {i+1}:\n"
        input_code += f"# Input ({input_grid.shape[0]}x{input_grid.shape[1]}):\n"
        input_code += f"# {convert_grid_to_string(input_grid).replace(chr(10), chr(10) + '# ')}\n"
        input_code += f"# Output ({output_grid.shape[0]}x{output_grid.shape[1]}):\n"
        input_code += f"# {convert_grid_to_string(output_grid).replace(chr(10), chr(10) + '# ')}\n"
    prompt = template.replace("{PUZZLE}", puzzle_text).replace("{INPUT_CODE}", input_code)
    return prompt


def test_program(code: str, task_data: dict, test_solutions: list = None) -> dict:
    """Test generated code on train AND test examples. Returns results dict."""
    results = {
        "train_passed": 0, "train_total": len(task_data["train"]),
        "test_passed": 0, "test_total": len(task_data.get("test", [])),
        "errors": []
    }

    # Test on train examples
    for i, ex in enumerate(task_data["train"]):
        input_grid = np.array(ex["input"], dtype=np.int8)
        expected_output = np.array(ex["output"], dtype=np.int8)

        try:
            exec_globals = {"np": np, "numpy": np, "input_grid": input_grid}
            full_code = code + "\n\nresult = generate_puzzle_output(input_grid)"
            exec(full_code, exec_globals)

            if "result" not in exec_globals:
                results["errors"].append(f"Train {i}: No result returned")
                continue

            actual_output = exec_globals["result"]
            validated = validate_and_convert_grid(actual_output)
            if validated is None:
                results["errors"].append(f"Train {i}: Invalid grid format")
                continue

            actual_array = np.array(validated, dtype=np.int8)
            if np.array_equal(actual_array, expected_output):
                results["train_passed"] += 1
            else:
                results["errors"].append(f"Train {i}: Output mismatch")
        except Exception as e:
            results["errors"].append(f"Train {i}: {type(e).__name__}: {str(e)[:50]}")

    # Test on test examples (if solutions available)
    if test_solutions:
        for i, ex in enumerate(task_data.get("test", [])):
            input_grid = np.array(ex["input"], dtype=np.int8)
            expected_output = np.array(test_solutions[i], dtype=np.int8)

            try:
                exec_globals = {"np": np, "numpy": np, "input_grid": input_grid}
                full_code = code + "\n\nresult = generate_puzzle_output(input_grid)"
                exec(full_code, exec_globals)

                if "result" not in exec_globals:
                    results["errors"].append(f"Test {i}: No result returned")
                    continue

                actual_output = exec_globals["result"]
                validated = validate_and_convert_grid(actual_output)
                if validated is None:
                    results["errors"].append(f"Test {i}: Invalid grid format")
                    continue

                actual_array = np.array(validated, dtype=np.int8)
                if np.array_equal(actual_array, expected_output):
                    results["test_passed"] += 1
                else:
                    results["errors"].append(f"Test {i}: Output mismatch")
            except Exception as e:
                results["errors"].append(f"Test {i}: {type(e).__name__}: {str(e)[:50]}")

    # Success = all train AND all test pass
    results["success"] = (results["train_passed"] == results["train_total"] and
                          results["test_passed"] == results["test_total"])
    return results


def generate_description(client: OpenAI, task_data: dict) -> dict:
    """Generate a description for a task. Returns dict with description and token usage."""
    prompt = format_description_prompt(task_data)
    response = call_gemini(client, prompt)

    description = recognize_summary(response["content"])
    return {
        "description": description,
        "input_tokens": response["input_tokens"],
        "output_tokens": response["output_tokens"],
        "thinking_tokens": response["thinking_tokens"],
    }


def generate_and_test_program(client: OpenAI, description: dict, task_data: dict, test_solutions: list, attempt: int) -> dict:
    """Generate output program and test it. Returns result dict."""
    prompt = format_output_program_prompt(description, task_data)
    response = call_gemini(client, prompt)

    code = parse_python_code(response["content"])
    if code is None:
        return {
            "success": False,
            "attempt": attempt,
            "error": "No Python code found in response",
            "input_tokens": response["input_tokens"],
            "output_tokens": response["output_tokens"],
            "thinking_tokens": response["thinking_tokens"],
        }

    test_results = test_program(code, task_data, test_solutions)
    return {
        "success": test_results["success"],
        "attempt": attempt,
        "code": code if test_results["success"] else None,
        "train_passed": test_results["train_passed"],
        "train_total": test_results["train_total"],
        "test_passed": test_results["test_passed"],
        "test_total": test_results["test_total"],
        "errors": test_results.get("errors", []),
        "input_tokens": response["input_tokens"],
        "output_tokens": response["output_tokens"],
        "thinking_tokens": response["thinking_tokens"],
    }


def validate_task(task_id: str, description_attempts: int = 2, output_program_attempts: int = 8, concurrent_requests: int = 4) -> dict:
    """Main validation loop for a single task."""
    # Load task
    with open(TASKS_FILE) as f:
        all_tasks = json.load(f)
    if task_id not in all_tasks:
        return {"task_id": task_id, "success": False, "error": f"Task {task_id} not found"}
    task_data = all_tasks[task_id]

    # Load test solutions
    with open(SOLUTIONS_FILE) as f:
        all_solutions = json.load(f)
    test_solutions = all_solutions.get(task_id, [])

    client = get_client()
    total_input = total_output = total_thinking = 0

    for desc_attempt in range(1, description_attempts + 1):
        print(f"\n=== Description attempt {desc_attempt}/{description_attempts} ===")

        # Generate description
        desc_prompt = format_description_prompt(task_data)
        desc_response = call_gemini(client, desc_prompt)
        total_input += desc_response["input_tokens"]
        total_output += desc_response["output_tokens"]
        total_thinking += desc_response["thinking_tokens"]

        description = recognize_summary(desc_response["content"])
        if description is None:
            print(f"Failed to parse description (attempt {desc_attempt})")
            continue

        print(f"Description generated. Rules: {description['rules_summary'][:100]}...")

        # Generate and test output programs concurrently
        print(f"Testing {output_program_attempts} output programs (concurrent={concurrent_requests})...")

        with ThreadPoolExecutor(max_workers=concurrent_requests) as executor:
            futures = {
                executor.submit(generate_and_test_program, client, description, task_data, test_solutions, i+1): i
                for i in range(output_program_attempts)
            }

            for future in as_completed(futures):
                result = future.result()
                total_input += result["input_tokens"]
                total_output += result["output_tokens"]
                total_thinking += result["thinking_tokens"]

                train_status = f"{result.get('train_passed', 0)}/{result.get('train_total', '?')}"
                test_status = f"{result.get('test_passed', 0)}/{result.get('test_total', '?')}"
                status = "PASS" if result["success"] else f"FAIL (train={train_status}, test={test_status})"
                errors = result.get("errors", [])
                error_str = f" - {errors[0]}" if errors else ""
                print(f"  Program {result['attempt']}: {status}{error_str}")

                if result["success"]:
                    # Cancel remaining futures
                    for f in futures:
                        f.cancel()

                    cost = (total_input * INPUT_PRICE / 1_000_000) + (total_output * OUTPUT_PRICE / 1_000_000)
                    return {
                        "task_id": task_id,
                        "success": True,
                        "description_attempt": desc_attempt,
                        "program_attempt": result["attempt"],
                        "validated_description": description,
                        "working_program": result["code"],
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
        "description_attempts_tried": description_attempts,
        "program_attempts_per_desc": output_program_attempts,
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "total_thinking_tokens": total_thinking,
        "total_cost_usd": round(cost, 4),
    }


def main():
    parser = argparse.ArgumentParser(description="Validate ARC task descriptions")
    parser.add_argument("--task", required=True, help="Task ID (e.g., 00576224)")
    parser.add_argument("--description-attempts", type=int, default=2, help="Number of description attempts (default: 2)")
    parser.add_argument("--output-program-attempts", type=int, default=8, help="Output programs per description (default: 8)")
    parser.add_argument("--concurrent-requests", type=int, default=4, help="Concurrent API calls (default: 4)")
    args = parser.parse_args()

    print(f"Validating task: {args.task}")
    print(f"Config: {args.description_attempts} desc attempts, {args.output_program_attempts} programs each, {args.concurrent_requests} concurrent")

    result = validate_task(
        args.task,
        description_attempts=args.description_attempts,
        output_program_attempts=args.output_program_attempts,
        concurrent_requests=args.concurrent_requests,
    )

    print("\n" + "=" * 60)
    print("RESULT:")
    print(json.dumps(result, indent=2))

    # Save result
    output_file = f"validation_result_{args.task}.json"
    with open(output_file, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to: {output_file}")


if __name__ == "__main__":
    main()
