# ARC-AGI-2 Task Validator & Trace Generation - Status

## Quick Start

```bash
# 1. Set your API key
export GEMINI_API_KEY="your-key-here"

# 2. Initialize submodule (if not done)
git submodule update --init external/ARC-AGI-2

# 3. Install dependencies
pip install aiohttp

# 4. Run trace generation (parallelized - ~2 min for 1000 tasks)
python scripts/generate_traces.py --concurrency 50

# 5. Resume if interrupted
python scripts/generate_traces.py --resume

# 6. Validate results
python scripts/validate_traces.py --report
```

## Overview

This project generates LLM traces for all 1000 ARC-AGI-2 training tasks using **Gemini Flash 2.0** to create seed data for the NVARC synthetic data pipeline.

## Current Status

| Phase | Status | Progress |
|-------|--------|----------|
| 1. Infrastructure Setup | Complete | 100% |
| 2. Trace Generation | Ready to Run | 0/1000 |
| 3. Validation & Quality Check | Ready | 0% |
| 4. Data Export | Not Started | 0% |

## Task Breakdown

### Phase 1: Infrastructure Setup
- [x] Initialize ARC-AGI-2 submodule (1000 training tasks available)
- [x] Verify GEMINI_API_KEY is set
- [x] Create parallelized trace generation script (`scripts/generate_traces.py`)
- [x] Create validation script (`scripts/validate_traces.py`)
- [x] Create output directories (`traces/raw/`, `traces/parsed/`)

### ⚠️ Note on Network Access
The current sandbox environment blocks external API calls. Run the scripts locally or in an environment with:
- Internet access to `generativelanguage.googleapis.com`
- Valid `GEMINI_API_KEY` exported

### Phase 2: Trace Generation (Parallelized)

**Strategy for Parallelization:**
- Use `asyncio` with configurable concurrency (default: 50 concurrent requests)
- Implement exponential backoff for rate limiting
- Save results incrementally (resume-capable)
- Track progress in real-time

**Output Format:**
Each task will produce a trace with:
- `rules_summary`: Transformation rules
- `input_generation`: Steps to generate input grids
- `solution_steps`: Steps to solve the puzzle
- `key_insight`: Core concept
- `puzzle_concepts`: Relevant concepts list

### Phase 3: Validation
- Parse all generated traces
- Verify required fields present
- Check for hallucinations/nonsense
- Generate quality report

## File Structure

```
NVARC/
├── traces/                    # Generated traces
│   ├── raw/                   # Raw Gemini responses
│   └── parsed/                # Validated structured traces
├── scripts/
│   ├── generate_traces.py     # Main parallelized generator
│   └── validate_traces.py     # Validation script
└── status.md                  # This file
```

## Parallelization Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Task Queue (1000 tasks)                       │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Async Semaphore (N=50)                        │
│  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐       ┌─────┐         │
│  │Task │ │Task │ │Task │ │Task │ │Task │  ...  │Task │         │
│  │ 1   │ │ 2   │ │ 3   │ │ 4   │ │ 5   │       │ 50  │         │
│  └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘       └──┬──┘         │
└─────┼──────┼──────┼──────┼──────┼────────────────┼─────────────┘
      │      │      │      │      │                │
      ▼      ▼      ▼      ▼      ▼                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Gemini Flash 2.0 API                          │
│                    (Rate-limited with backoff)                   │
└─────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Progress Tracker                              │
│  - Completed: XXX/1000                                           │
│  - Failed: XX                                                    │
│  - Remaining: XXX                                                │
└─────────────────────────────────────────────────────────────────┘
```

## Estimated Time

With 50 concurrent requests and ~2-3 seconds per request:
- Sequential: ~50 minutes (1000 * 3s)
- Parallel (50x): **~1-2 minutes**

## Commands

```bash
# Generate traces (parallelized)
python scripts/generate_traces.py --concurrency 50

# Resume interrupted generation
python scripts/generate_traces.py --resume

# Validate generated traces
python scripts/validate_traces.py

# Check progress
python scripts/generate_traces.py --status
```

## Next Steps

1. Create `scripts/generate_traces.py` with async parallelization
2. Test with 10 tasks first
3. Run full 1000-task generation
4. Validate and create quality report
