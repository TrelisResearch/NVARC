#!/usr/bin/env python3
"""
Validate generated traces and produce quality reports.

Usage:
    python scripts/validate_traces.py           # Validate all traces
    python scripts/validate_traces.py --report  # Generate detailed report
"""

import argparse
import json
from pathlib import Path
from collections import Counter

PARSED_DIR = Path("traces/parsed")
RAW_DIR = Path("traces/raw")
TRAINING_DIR = Path("external/ARC-AGI-2/data/training")

REQUIRED_FIELDS = ["rules_summary", "solution_steps"]
OPTIONAL_FIELDS = ["input_generation", "key_insight", "puzzle_concepts"]


def validate_trace(trace: dict) -> tuple[bool, list[str]]:
    """Validate a single trace, returning (valid, issues)."""
    issues = []

    # Check required fields
    for field in REQUIRED_FIELDS:
        if field not in trace:
            issues.append(f"Missing required field: {field}")
        elif not trace[field] or len(trace[field].strip()) < 10:
            issues.append(f"Field '{field}' too short or empty")

    # Check optional fields
    for field in OPTIONAL_FIELDS:
        if field not in trace:
            issues.append(f"Missing optional field: {field}")

    # Quality checks
    if "rules_summary" in trace:
        if len(trace["rules_summary"]) < 50:
            issues.append("rules_summary too brief")

    if "solution_steps" in trace:
        if "1." not in trace["solution_steps"]:
            issues.append("solution_steps may not be properly numbered")

    return len([i for i in issues if "required" in i.lower()]) == 0, issues


def run_validation(generate_report: bool = False):
    """Run validation on all parsed traces."""
    all_tasks = set(f.stem for f in TRAINING_DIR.glob("*.json"))
    parsed_traces = list(PARSED_DIR.glob("*.json"))
    raw_responses = list(RAW_DIR.glob("*.md"))

    print("ARC-AGI-2 Trace Validation Report")
    print("=" * 50)

    # Coverage stats
    parsed_ids = set(f.stem for f in parsed_traces)
    raw_ids = set(f.stem for f in raw_responses)

    print(f"\nCoverage:")
    print(f"  Total tasks: {len(all_tasks)}")
    print(f"  Raw responses: {len(raw_ids)}")
    print(f"  Parsed traces: {len(parsed_ids)}")
    print(f"  Missing: {len(all_tasks - raw_ids)}")

    # Validate each trace
    valid_count = 0
    invalid_count = 0
    all_issues = []
    field_stats = Counter()

    for trace_file in parsed_traces:
        with open(trace_file) as f:
            trace = json.load(f)

        valid, issues = validate_trace(trace)

        if valid:
            valid_count += 1
        else:
            invalid_count += 1

        # Track field presence
        for field in REQUIRED_FIELDS + OPTIONAL_FIELDS:
            if field in trace and trace[field]:
                field_stats[field] += 1

        if issues:
            all_issues.append((trace_file.stem, issues))

    print(f"\nValidation Results:")
    print(f"  Valid traces: {valid_count}")
    print(f"  Invalid traces: {invalid_count}")

    print(f"\nField Presence (of {len(parsed_traces)} parsed):")
    for field in REQUIRED_FIELDS + OPTIONAL_FIELDS:
        count = field_stats[field]
        pct = (count / len(parsed_traces) * 100) if parsed_traces else 0
        marker = "*" if field in REQUIRED_FIELDS else " "
        print(f"  {marker}{field}: {count} ({pct:.1f}%)")

    if generate_report and all_issues:
        print(f"\nIssues by Task (showing first 20):")
        for task_id, issues in all_issues[:20]:
            print(f"\n  {task_id}:")
            for issue in issues:
                print(f"    - {issue}")

    # Quality summary
    if parsed_traces:
        quality_score = valid_count / len(parsed_traces) * 100
        print(f"\nOverall Quality Score: {quality_score:.1f}%")


def main():
    parser = argparse.ArgumentParser(description="Validate ARC-AGI-2 traces")
    parser.add_argument("--report", action="store_true", help="Generate detailed report")
    args = parser.parse_args()

    run_validation(generate_report=args.report)


if __name__ == "__main__":
    main()
