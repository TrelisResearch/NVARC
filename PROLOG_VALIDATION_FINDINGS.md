# Prolog Validation Findings

## Summary

Investigation into using Prolog for ARC task validation with LLM-generated recognizers and transforms.

## Key Discovery: Checker-Style vs Generator-Style

**Checker-style Prolog works reliably; Generator-style (CLP(FD)) fails consistently.**

| Style | Purpose | Success Rate | Notes |
|-------|---------|--------------|-------|
| Checker | Validation only | ~100% (1/1 tested) | Simple syntax, LLM generates correct code |
| Generator | Validation + Generation | ~0% (0/25 attempts) | CLP(FD) syntax errors, over-constrained |

### Checker-Style (Recommended)
- Uses standard Prolog: `==`, `\==`, `>=`, `=<`
- Only validates inputs (accepts/rejects)
- LLM generates correct code on first attempt
- Example task `aa18de87`: Train 4/4, Test 1/1, Specificity 10/10 ✓

### Generator-Style (CLP(FD))
- Uses constraint operators: `#=`, `#\=`, `#>=`, `ins`
- Can generate valid inputs (for synthetic data)
- LLM struggles with syntax, produces broken code
- 0/25 attempts succeeded across 5 tasks

## Key Changes Made

### 1. Added Prolog Query Timeouts
- **File**: `validate_prolog_task.py`
- **Problem**: Generated Prolog code could have infinite loops, causing tests to hang indefinitely
- **Solution**: Added 5-second timeout using Python's `signal.SIGALRM`
- **Result**: Tests now complete in ~5 minutes instead of hanging

### 2. Added CLP(FD) Examples to Prompt
- **File**: `SDG/prompts/prolog_input_recognizer.md`
- **Problem**: LLM was making syntax errors with CLP(FD) operators (`#=`, `#\=`, `ins`, etc.)
- **Solution**: Added operator reference table and examples showing CLP(FD) vs checker-style syntax
- **Result**: Generated code now correctly uses CLP(FD) syntax

### 3. Fixed PySwip Thread Safety Issue
- **File**: `validate_prolog_task.py`
- **Problem**: PySwip uses a single global Prolog process, causing race conditions with concurrent threads
- **Solution**: Added global `_prolog_lock` around all Prolog operations in `test_input_recognizer()` and `test_transform_full()`
- **Result**: Thread-safe execution, but Prolog operations are now serialized

## Current Status

### What Works
- Timeouts prevent hangs
- Thread-safe Prolog operations
- Specificity tests pass (recognizers correctly reject random grids)
- Single-task runs occasionally produce working recognizers (e.g., `aa18de87` showed Train: 4/4)

### What Doesn't Work
- **0% success rate in batch runs**: 25 recognizer attempts across 5 tasks all failed (train=0)
- **High LLM code quality variance**: Same task produces different code quality each API call
- **Generalization issues**: Some recognizers pass train but fail test inputs

## Root Cause Analysis

The core issue is **LLM-generated Prolog code quality variance**:

1. **Different algorithms each time**: The LLM generates different validation logic each API call
2. **Some algorithms are wrong**: For example, one generated recognizer expected columns with no colored cells, but the actual puzzle had colored cells in every column
3. **Over-constrained recognizers**: Many generated recognizers are too specific and reject even the training examples they were shown

### Example: Task aa18de87

**Good code** (passed train=4/4):
- Used modulo 8 pattern with K1, K2 parameters
- Correctly captured the V-shaped diagonal pattern

**Bad code** (failed train=0/4):
- Used `extract_path` expecting columns with no colored cells
- Algorithm didn't match actual puzzle structure

## Recommendations

### Short Term
1. **More attempts**: Increase recognizer rounds (current: 2, suggest: 10+)
2. **Better prompts**: Add more worked examples of different pattern types
3. **Temperature tuning**: Lower temperature for more consistent code generation

### Medium Term
1. **Code validation**: Add static analysis to reject obviously broken code before Prolog execution
2. **Feedback loop**: Pass error messages back to LLM for self-correction
3. **Pattern library**: Pre-define common patterns (diagonal, horizontal, symmetric) as Prolog templates

### Long Term
1. **Different approach**: Consider DSL or structured output instead of free-form Prolog
2. **Replace pyswip with subprocess**: Use `brew install swi-prolog` and call via subprocess instead of pyswip. Each subprocess is isolated, avoiding the single global Prolog process limitation that breaks parallelism.
3. **Alternative Prolog**: Consider tau-prolog (JavaScript) or other isolated implementations

## Files

### New Files
| File | Purpose |
|------|---------|
| `validate_prolog_single.py` | Single-task validation with `--style checker/generator` flag |
| `SDG/prompts/prolog_input_recognizer_checker.md` | Checker-style prompt (simple Prolog) |
| `SDG/prompts/prolog_input_recognizer_generator.md` | Generator-style prompt (CLP(FD)) |

### Modified Files
| File | Changes |
|------|---------|
| `validate_prolog_task.py` | Added timeouts, thread lock |
| `validate_prolog_all.py` | Minor cleanup |
| `SDG/prompts/prolog_input_recognizer.md` | Added CLP(FD) operator reference |
| `llm_utils.py` | Added evaluation task file paths |

## Test Results

| Test | Tasks | Recognizer Rounds | Success Rate |
|------|-------|-------------------|--------------|
| Baseline (checker-style) | 5 | 2 | 0% |
| With CLP(FD) examples | 5 | 2 | 0% |
| With thread safety fix | 5 | 5 | 0% |
| Single task (aa18de87) | 1 | 1 | ~50% (varies) |

## Cost Analysis

- ~$0.004-0.005 per task per recognizer attempt
- ~$0.02 for 5 tasks with 2 rounds
- Estimated ~$4-5 for full 1000-task training set with 2 rounds
