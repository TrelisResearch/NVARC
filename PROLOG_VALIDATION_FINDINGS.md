# Prolog Validation Findings

## Summary

Investigation into using Prolog for ARC task validation with LLM-generated recognizers and transforms.

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
2. **Multiprocessing**: Replace threading with multiprocessing for true Prolog isolation
3. **Alternative Prolog**: Consider tau-prolog (JavaScript) or other isolated implementations

## Files Modified

| File | Changes |
|------|---------|
| `validate_prolog_task.py` | Added timeouts, thread lock, debug output |
| `validate_prolog_all.py` | Debug output (removed) |
| `SDG/prompts/prolog_input_recognizer.md` | Added CLP(FD) operator reference |

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
