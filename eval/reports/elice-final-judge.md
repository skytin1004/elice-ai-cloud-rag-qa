# Eval Report: elice-final-judge

## Provider

- Generated at: `2026-06-27T14:30:47.720433+00:00`
- LLM provider: `elice`
- LLM model/deployment: `openai/gpt-5-mini`
- Embedding provider: `elice`
- Embedding model: `openai/text-embedding-3-small`
- Top-k: `5`
- Min score: `0.08`
- Git commit: `731b41d`
- Git branch: `main`
- Python: `3.12.6`
- Seed policy: `not used; deterministic retrieval/eval path`
- Judge policy: `enabled; fixed rubric JSON judge; temperature=0.0; model=openai/gpt-5-mini; max_context_chars=6000`

## Summary

| Metric | Score |
|---|---:|
| Retrieval recall@k | 0.9444 |
| Citation hit rate | 0.8889 |
| Refusal accuracy | 1.0000 |
| Faithfulness | 0.9250 |
| Judge groundedness | 0.8500 |
| Judge correctness | 0.6750 |
| Judge score | 0.7625 |
| Judge pass rate | 0.6000 |

## Examples

| ID | Status | Retrieval | Citation | Refusal | Faithfulness | Judge | Judge pass |
|---|---|---:|---:|---:|---:|---:|---:|
| q001 | answered | True | True |  | 1.00 | 0.75 | False |
| q002 | answered | True | True |  | 1.00 | 1.00 | True |
| q003 | answered | True | True |  | 1.00 | 0.50 | False |
| q004 | answered | True | True |  | 1.00 | 1.00 | True |
| q005 | answered | True | True |  | 1.00 | 0.00 | False |
| q006 | insufficient_context | True | False |  | 0.00 | 0.00 | False |
| q007 | answered | True | True |  | 1.00 | 0.75 | False |
| q008 | answered | True | True |  | 1.00 | 1.00 | True |
| q009 | answered | True | True |  | 1.00 | 1.00 | True |
| q010 | answered | True | True |  | 1.00 | 0.50 | False |
| q011 | answered | True | True |  | 1.00 | 0.25 | False |
| q012 | answered | True | True |  | 1.00 | 1.00 | True |
| q013 | answered | False | False |  | 0.50 | 1.00 | True |
| q014 | answered | True | True |  | 1.00 | 1.00 | True |
| q015 | answered | True | True |  | 1.00 | 1.00 | True |
| q016 | answered | True | True |  | 1.00 | 0.50 | False |
| q017 | answered | True | True |  | 1.00 | 1.00 | True |
| q018 | answered | True | True |  | 1.00 | 1.00 | True |
| q019 | insufficient_context | False | False | True | 1.00 | 1.00 | True |
| q020 | insufficient_context | False | False | True | 1.00 | 1.00 | True |
