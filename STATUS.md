# ARC Task Validator - Status

## Completed
- `validate_task.py` - Generates descriptions via Gemini 3 Flash, tests Python output programs
- `validate_all_tasks.py` - Batched parallel Python validation
- `validate_prolog_task.py` - Prolog-based recognizer + transform validation
- `validate_prolog_all.py` - Batched parallel Prolog validation
- Downloaded 1000 training tasks + solutions to `arc_agi2_training_only/`

## Python Validation Results (64 tasks)
- Success: 53/64 (82.8%)
- Cost: $0.67 (~$0.01/task)
- Time: 20.1 min

## Prolog Validation Results (10 tasks)
- Success: 5/10 (50%)
- Cost: $0.066 (~$0.007/task)
- Time: 7.6 min

## Architecture

### Python Pipeline (validate_all_tasks.py)
```
ROUND 1:
  Description Phase (64 concurrent) → 1 description per task
  Program Phase (64 concurrent) → up to 4 attempts until success

ROUND 2 (failures only):
  New descriptions → retry programs
```

### Prolog Pipeline (validate_prolog_all.py)
Two-phase approach using Prolog predicates:
```
ROUND 1:
  Recognizer Phase → Generate valid_input/1 predicate
    - Validate on train + test inputs
    - Specificity test: must reject >= 90% random grids
  Transform Phase → Generate transform/2 predicate
    - Validate on train + test pairs (exact match)

ROUND 2 (failures only):
  New recognizers → new transforms
```

## CLI Arguments

### Python Validator
```
uv run validate_all_tasks.py --sample 10 --max-concurrent 64
```

### Prolog Validator
```
uv run validate_prolog_all.py --sample 10 --max-concurrent 32
```

## Dependencies
- `pyswip` - Python bindings for SWI-Prolog
- SWI-Prolog (`brew install swi-prolog`)

## Files
- `validation_results.jsonl` - Python validation results
- `prolog_validation_results.jsonl` - Prolog validation results
- `SDG/prompts/prolog_input_recognizer.md` - Prompt for valid_input/1
- `SDG/prompts/prolog_transform.md` - Prompt for transform/2
