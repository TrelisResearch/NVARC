"""
Shared Prolog utilities for ARC task validation.

Uses subprocess for cross-platform timeout and thread safety.
"""
import json
import os
import re
import subprocess
import sys
import tempfile


# Timeout for Prolog queries (seconds)
PROLOG_TIMEOUT = 5


class PrologTimeoutError(Exception):
    """Raised when a Prolog query exceeds the timeout."""
    pass


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


def run_prolog_query(code: str, query: str, timeout_sec: int = PROLOG_TIMEOUT) -> dict:
    """Run a Prolog query in an isolated subprocess.

    Args:
        code: Prolog code to consult
        query: Query to run (e.g., "valid_input([[1,2],[3,4]])")
        timeout_sec: Timeout in seconds

    Returns:
        dict with keys:
        - success: bool
        - results: list of results (empty if failed)
        - error: error message if failed
    """
    # Write code to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.pl', delete=False) as f:
        f.write(code)
        temp_path = f.name

    # Python script to run in subprocess
    script = f'''
import sys
try:
    from pyswip import Prolog
    prolog = Prolog()
    prolog.consult("{temp_path}")
    results = list(prolog.query("{query}", maxresult=1))
    print("SUCCESS")
    print(len(results))
except Exception as e:
    print("ERROR")
    print(str(e)[:200])
'''

    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=timeout_sec
        )

        lines = result.stdout.strip().split('\n')
        if lines and lines[0] == "SUCCESS":
            count = int(lines[1]) if len(lines) > 1 else 0
            return {"success": True, "results": [{}] * count, "error": None}
        else:
            error = lines[1] if len(lines) > 1 else result.stderr[:200]
            return {"success": False, "results": [], "error": error}

    except subprocess.TimeoutExpired:
        return {"success": False, "results": [], "error": "timeout"}
    except Exception as e:
        return {"success": False, "results": [], "error": str(e)[:200]}
    finally:
        try:
            os.unlink(temp_path)
        except OSError:
            pass


def test_valid_input(code: str, grid: list, timeout_sec: int = PROLOG_TIMEOUT) -> bool:
    """Test if valid_input(Grid) succeeds for the given grid.

    Args:
        code: Prolog code containing valid_input/1 predicate
        grid: Grid to test
        timeout_sec: Timeout in seconds

    Returns:
        True if valid_input succeeds, False otherwise

    Raises:
        PrologTimeoutError if query times out
    """
    grid_str = grid_to_prolog(grid)
    query = f"valid_input({grid_str})"

    result = run_prolog_query(code, query, timeout_sec)

    if result["error"] == "timeout":
        raise PrologTimeoutError("Prolog query timed out")

    return result["success"] and len(result["results"]) > 0


def test_transform(code: str, input_grid: list, expected_output: list, timeout_sec: int = PROLOG_TIMEOUT) -> bool:
    """Test if transform(Input, Output) produces the expected output.

    Args:
        code: Prolog code containing transform/2 predicate
        input_grid: Input grid
        expected_output: Expected output grid
        timeout_sec: Timeout in seconds

    Returns:
        True if transform produces expected output, False otherwise

    Raises:
        PrologTimeoutError if query times out
    """
    input_str = grid_to_prolog(input_grid)
    expected_str = grid_to_prolog(expected_output)
    query = f"transform({input_str}, Output), Output = {expected_str}"

    result = run_prolog_query(code, query, timeout_sec)

    if result["error"] == "timeout":
        raise PrologTimeoutError("Prolog query timed out")

    return result["success"] and len(result["results"]) > 0
