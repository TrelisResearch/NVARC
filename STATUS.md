# ARC Task Validator - Status

## Completed
- `validate_task.py` - Generates descriptions via Gemini 3 Flash, tests output programs on train + test examples
- `validate_all_tasks.py` - Batched parallel execution with separate description and program phases
- Downloaded 1000 training tasks + solutions to `arc_agi2_training_only/`

## Latest Results (10 tasks, batched parallel)
- Success: 10/10 (100%)
- Cost: $0.093 (~$0.009/task)
- Time: 15.5 min
- Avg time/task: 93s

## Architecture
Batched pipeline with single pool per phase:
```
ROUND 1:
  Description Phase (64 concurrent) → 1 description per task
  Program Phase (64 concurrent) → up to 4 attempts until success

ROUND 2 (failures only):
  New descriptions → retry programs
```

## CLI Arguments
```
--max-concurrent N          Max concurrent API calls (default: 64)
--max-description-rounds N  Max description attempts per task (default: 2)
--max-program-attempts N    Max program attempts per description (default: 4)
--sample N                  Run on N random tasks only
--clear                     Clear previous results
```

## Expected Performance
- 1000 tasks with 64 concurrent: ~26 hours
- Can increase concurrency if API allows

## Next Steps
1. **Run full 1000 tasks** with batched pipeline

## Files
- `validation_results.jsonl` - Individual task results
- `validation_summary.json` - Aggregate statistics
