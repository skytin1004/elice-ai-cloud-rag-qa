# Eval Report: improved-heading-threshold

## Provider

- Generated at: `2026-06-27T08:17:47.737978+00:00`
- LLM provider: `elice`
- LLM model/deployment: `openai/gpt-5-mini`
- Embedding provider: `elice`
- Embedding model: `openai/text-embedding-3-small`
- Top-k: `5`
- Min score: `0.08`
- Git commit: `5a968a5`
- Git branch: `main`
- Python: `3.12.6`
- Seed policy: `not used; deterministic retrieval/eval path`
- Judge policy: `not used; deterministic metric implementation`

## Summary

| Metric | Score |
|---|---:|
| Retrieval recall@k | 0.9444 |
| Citation hit rate | 0.8889 |
| Refusal accuracy | 1.0000 |
| Faithfulness | 0.9250 |

## Examples

| ID | Status | Retrieval | Citation | Refusal | Faithfulness |
|---|---|---:|---:|---:|---:|
| q001 | answered | True | True |  | 1.00 |
| q002 | answered | True | True |  | 1.00 |
| q003 | answered | True | True |  | 1.00 |
| q004 | answered | True | True |  | 1.00 |
| q005 | answered | True | True |  | 1.00 |
| q006 | insufficient_context | True | False |  | 0.00 |
| q007 | answered | True | True |  | 1.00 |
| q008 | answered | True | True |  | 1.00 |
| q009 | answered | True | True |  | 1.00 |
| q010 | answered | True | True |  | 1.00 |
| q011 | answered | True | True |  | 1.00 |
| q012 | answered | True | True |  | 1.00 |
| q013 | answered | False | False |  | 0.50 |
| q014 | answered | True | True |  | 1.00 |
| q015 | answered | True | True |  | 1.00 |
| q016 | answered | True | True |  | 1.00 |
| q017 | answered | True | True |  | 1.00 |
| q018 | answered | True | True |  | 1.00 |
| q019 | insufficient_context | False | False | True | 1.00 |
| q020 | insufficient_context | False | False | True | 1.00 |
