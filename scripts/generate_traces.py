#!/usr/bin/env python3
"""
Parallelized trace generation for ARC-AGI-2 training tasks using Gemini Flash 2.0.

Usage:
    python scripts/generate_traces.py                    # Run with defaults (50 concurrent)
    python scripts/generate_traces.py --concurrency 100  # Higher concurrency
    python scripts/generate_traces.py --resume           # Resume interrupted run
    python scripts/generate_traces.py --status           # Check progress
    python scripts/generate_traces.py --limit 10         # Process only first 10 tasks
"""

import asyncio
import argparse
import json
import os
import sys
import time
import re
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import aiohttp

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from llm_utils import MODEL

# Configuration
TRAINING_DIR = Path("external/ARC-AGI-2/data/training")
RAW_OUTPUT_DIR = Path("traces/raw")
PARSED_OUTPUT_DIR = Path("traces/parsed")
PROGRESS_FILE = Path("traces/progress.json")

# Gemini configuration (use model from llm_utils for consistency)
MODEL_NAME = MODEL
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent"
MAX_RETRIES = 3
INITIAL_BACKOFF = 1.0  # seconds

# Color mapping for grids
COLORS = {
    0: "BLACK", 1: "BLUE", 2: "RED", 3: "GREEN", 4: "YELLOW",
    5: "GRAY", 6: "MAGENTA", 7: "ORANGE", 8: "SKY", 9: "BROWN"
}

PROMPT_TEMPLATE = """You are an expert at solving ARC-AGI puzzles. These puzzles involve discovering hidden rules that transform input grids into output grids.

Analyze the following puzzle with its training examples. Each grid is a 2D array of colored cells (0-9):
Colors: 0=BLACK, 1=BLUE, 2=RED, 3=GREEN, 4=YELLOW, 5=GRAY, 6=MAGENTA, 7=ORANGE, 8=SKY, 9=BROWN

{examples}

Your task is to analyze this puzzle and provide a comprehensive trace of your reasoning. Follow these steps:

1. Carefully observe each input-output pair
2. Identify patterns, transformations, and relationships
3. Formulate the rule that transforms inputs to outputs
4. Describe how to generate similar input grids
5. List the key concepts used in this puzzle

Provide your analysis in the following format:

<rules_summary>
Concise summary of the transformation rules that convert input to output.
</rules_summary>

<input_generation>
1. [First step to generate valid input grids for this puzzle type]
2. [Second step...]
3. [Continue as needed]
</input_generation>

<solution_steps>
1. [First step to apply the transformation]
2. [Second step...]
3. [Continue as needed]
</solution_steps>

<key_insight>
The core concept or main insight required to solve this puzzle.
</key_insight>

<puzzle_concepts>
- [concept 1]
- [concept 2]
- [continue as needed]
</puzzle_concepts>

Be specific and precise. Focus on the actual transformation patterns you observe."""


@dataclass
class TaskResult:
    task_id: str
    success: bool
    raw_response: Optional[str] = None
    parsed_trace: Optional[dict] = None
    error: Optional[str] = None
    duration: float = 0.0


def grid_to_string(grid: list) -> str:
    """Convert a 2D grid to a formatted string representation."""
    lines = []
    for row in grid:
        lines.append(" ".join(str(cell) for cell in row))
    return "\n".join(lines)


def format_examples(task_data: dict) -> str:
    """Format training examples for the prompt."""
    examples = []
    for i, example in enumerate(task_data["train"], 1):
        input_grid = grid_to_string(example["input"])
        output_grid = grid_to_string(example["output"])
        examples.append(f"""### Training Example {i}

**Input Grid:**
```
{input_grid}
```

**Output Grid:**
```
{output_grid}
```
""")
    return "\n".join(examples)


def parse_trace(response: str) -> Optional[dict]:
    """Parse the structured trace from Gemini response."""
    patterns = {
        "rules_summary": r"<rules_summary>(.*?)</rules_summary>",
        "input_generation": r"<input_generation>(.*?)</input_generation>",
        "solution_steps": r"<solution_steps>(.*?)</solution_steps>",
        "key_insight": r"<key_insight>(.*?)</key_insight>",
        "puzzle_concepts": r"<puzzle_concepts>(.*?)</puzzle_concepts>",
    }

    result = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, response, re.DOTALL)
        if match:
            result[key] = match.group(1).strip()

    # Require at least rules_summary and solution_steps
    if "rules_summary" in result and "solution_steps" in result:
        return result
    return None


def load_progress() -> dict:
    """Load progress from file."""
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {"completed": [], "failed": [], "start_time": None}


def save_progress(progress: dict):
    """Save progress to file."""
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f, indent=2)


def get_all_tasks() -> list[str]:
    """Get all task IDs from training directory."""
    tasks = []
    for f in TRAINING_DIR.glob("*.json"):
        tasks.append(f.stem)
    return sorted(tasks)


async def process_task(
    task_id: str,
    session: aiohttp.ClientSession,
    api_key: str,
    semaphore: asyncio.Semaphore,
    progress: dict
) -> TaskResult:
    """Process a single task with the Gemini API."""
    start_time = time.time()

    async with semaphore:
        # Load task data
        task_file = TRAINING_DIR / f"{task_id}.json"
        with open(task_file) as f:
            task_data = json.load(f)

        # Build prompt
        examples = format_examples(task_data)
        prompt = PROMPT_TEMPLATE.format(examples=examples)

        # API request body
        request_body = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.7,
                "maxOutputTokens": 4096,
            }
        }

        url = f"{API_URL}?key={api_key}"

        # Call Gemini with retries
        for attempt in range(MAX_RETRIES):
            try:
                async with session.post(url, json=request_body) as response:
                    if response.status == 429:
                        # Rate limited
                        backoff = INITIAL_BACKOFF * (2 ** attempt)
                        await asyncio.sleep(backoff)
                        continue

                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"API error {response.status}: {error_text[:200]}")

                    result = await response.json()

                # Extract text from response
                raw_text = result["candidates"][0]["content"]["parts"][0]["text"]

                # Save raw response
                raw_file = RAW_OUTPUT_DIR / f"{task_id}.md"
                with open(raw_file, "w") as f:
                    f.write(raw_text)

                # Parse trace
                parsed = parse_trace(raw_text)

                if parsed:
                    # Save parsed trace
                    parsed_file = PARSED_OUTPUT_DIR / f"{task_id}.json"
                    with open(parsed_file, "w") as f:
                        json.dump(parsed, f, indent=2)

                    return TaskResult(
                        task_id=task_id,
                        success=True,
                        raw_response=raw_text,
                        parsed_trace=parsed,
                        duration=time.time() - start_time
                    )
                else:
                    return TaskResult(
                        task_id=task_id,
                        success=False,
                        raw_response=raw_text,
                        error="Failed to parse trace structure",
                        duration=time.time() - start_time
                    )

            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    backoff = INITIAL_BACKOFF * (2 ** attempt)
                    await asyncio.sleep(backoff)
                else:
                    return TaskResult(
                        task_id=task_id,
                        success=False,
                        error=str(e),
                        duration=time.time() - start_time
                    )


async def run_generation(
    concurrency: int = 50,
    limit: Optional[int] = None,
    resume: bool = False
):
    """Run parallelized trace generation."""
    # Get API key
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY not set")
        sys.exit(1)

    # Get tasks
    all_tasks = get_all_tasks()

    # Load progress for resume
    progress = load_progress() if resume else {"completed": [], "failed": [], "start_time": None}

    if progress["start_time"] is None:
        progress["start_time"] = time.time()

    # Filter out completed tasks if resuming
    if resume:
        completed_set = set(progress["completed"])
        tasks = [t for t in all_tasks if t not in completed_set]
        print(f"Resuming: {len(progress['completed'])} completed, {len(tasks)} remaining")
    else:
        tasks = all_tasks

    if limit:
        tasks = tasks[:limit]

    print(f"Processing {len(tasks)} tasks with concurrency={concurrency}")
    print(f"Model: {MODEL_NAME}")
    print("-" * 60)

    # Create semaphore for concurrency control
    semaphore = asyncio.Semaphore(concurrency)

    # Process all tasks
    completed = 0
    failed = 0
    start_time = time.time()

    # Create aiohttp session with connection pool
    connector = aiohttp.TCPConnector(limit=concurrency + 10)
    timeout = aiohttp.ClientTimeout(total=60)

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        # Create tasks
        async_tasks = [
            process_task(task_id, session, api_key, semaphore, progress)
            for task_id in tasks
        ]

        # Run with progress tracking
        for coro in asyncio.as_completed(async_tasks):
            result = await coro

            if result.success:
                completed += 1
                progress["completed"].append(result.task_id)
            else:
                failed += 1
                progress["failed"].append({
                    "task_id": result.task_id,
                    "error": result.error
                })

            # Progress update
            total_done = completed + failed
            elapsed = time.time() - start_time
            rate = total_done / elapsed if elapsed > 0 else 0
            remaining = len(tasks) - total_done
            eta = remaining / rate if rate > 0 else 0

            print(f"\r[{total_done}/{len(tasks)}] "
                  f"Completed: {completed} | Failed: {failed} | "
                  f"Rate: {rate:.1f}/s | ETA: {eta:.0f}s", end="", flush=True)

            # Save progress periodically
            if total_done % 10 == 0:
                save_progress(progress)

    # Final save
    save_progress(progress)

    print(f"\n{'=' * 60}")
    print(f"Generation complete!")
    print(f"  Completed: {completed}")
    print(f"  Failed: {failed}")
    print(f"  Total time: {time.time() - start_time:.1f}s")


def show_status():
    """Show current progress status."""
    progress = load_progress()
    all_tasks = get_all_tasks()

    print("ARC-AGI-2 Trace Generation Status")
    print("=" * 40)
    print(f"Total tasks: {len(all_tasks)}")
    print(f"Completed: {len(progress.get('completed', []))}")
    print(f"Failed: {len(progress.get('failed', []))}")
    print(f"Remaining: {len(all_tasks) - len(progress.get('completed', []))}")

    if progress.get("failed"):
        print(f"\nFailed tasks:")
        for f in progress["failed"][:5]:
            print(f"  - {f['task_id']}: {f['error'][:50]}...")


def main():
    parser = argparse.ArgumentParser(description="Generate traces for ARC-AGI-2 tasks")
    parser.add_argument("--concurrency", type=int, default=50, help="Number of concurrent requests")
    parser.add_argument("--limit", type=int, help="Limit number of tasks to process")
    parser.add_argument("--resume", action="store_true", help="Resume from last run")
    parser.add_argument("--status", action="store_true", help="Show progress status")

    args = parser.parse_args()

    if args.status:
        show_status()
        return

    # Ensure output directories exist
    RAW_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PARSED_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Run async generation
    asyncio.run(run_generation(
        concurrency=args.concurrency,
        limit=args.limit,
        resume=args.resume
    ))


if __name__ == "__main__":
    main()
