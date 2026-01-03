# ARC Task Validator - Remaining Plan

## Goal
Validate ARC-AGI-2 task descriptions by generating output programs and testing them against original train + test examples (Option B - no synthetic input generation).

## Completed
- [x] Branch: `arc-task-validator` off `main`
- [x] Tested Gemini 3 Flash APIs (native + OpenAI-compatible)
- [x] Verified thinking token tracking works
- [x] `test_gemini_apis.py` committed

## Next: Create `validate_task.py`

### Flow
```
1. Load task from arc_agi2_training_only/arc-agi_training_challenges.json
2. For each description attempt (default 2):
   a. Format summary_v2.md prompt with all train examples
   b. Call Gemini 3 Flash → get structured description
   c. For each output program attempt (default 8, concurrent):
      i. Format generate_puzzle_output.md with description + train examples as reference
      ii. Call Gemini 3 Flash → get Python code
      iii. Test on ALL train inputs → check outputs match
      iv. Test on ALL test inputs → check outputs match
      v. If ALL pass → SUCCESS
3. Output results JSON with costs
```

### CLI Interface
```bash
uv run validate_task.py --task 00576224
uv run validate_task.py --task 00576224 --description-attempts 2 --output-program-attempts 8
```

### Defaults
- `--description-attempts=2`
- `--output-program-attempts=8`
- `--concurrent-requests=4`
- `--reasoning-effort=high`

### API Choice
Use **OpenAI-compatible API** for model flexibility:
```python
from openai import OpenAI
client = OpenAI(
    api_key=os.environ.get("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)
response = client.chat.completions.create(
    model="gemini-3-flash-preview",
    messages=[{"role": "user", "content": prompt}],
    extra_body={"reasoning_effort": "high"},
)
```

### Cost Tracking
- Input: $0.50 / 1M tokens
- Output: $3.00 / 1M tokens
- Thinking tokens: inferred from `total - prompt - completion`

### Prompt Formatting
1. **summary_v2.md**: Replace `{PUZZLE}` with train examples, skip `{EXAMPLES}` (zero-shot)
2. **generate_puzzle_output.md**: Replace `{PUZZLE}` with description, `{INPUT_CODE}` with train examples as reference

### Reuse from SDG/scripts/
- `parser.py`: `parse_python_code()` - extract code from markdown
- `puzzle.py`: `execute_code()`, `validate_and_convert_grid()` - safe execution
- `utils.py`: `recognize_summary()` - parse description

### Output Format
```json
{
  "task_id": "00576224",
  "success": true,
  "description_attempt": 1,
  "program_attempt": 3,
  "validated_description": { ... },
  "working_program": "def generate_puzzle_output(input_grid):\n    ...",
  "train_passed": 3,
  "test_passed": 1,
  "total_input_tokens": 15234,
  "total_output_tokens": 4521,
  "total_thinking_tokens": 8932,
  "total_cost_usd": 0.0212
}
```

## Test Task
Start with task `00576224` (first in the dataset).
