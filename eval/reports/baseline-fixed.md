# Eval Report: baseline-fixed-top3

## Provider

- Generated at: `2026-06-26T05:32:36.848786+00:00`
- LLM provider: `mock`
- LLM model/deployment: `extractive-mock-v1`
- Embedding provider: `local`
- Embedding model: `hashing-ngrams-v1`
- Top-k: `3`
- Min score: `0.0`
- Git commit: `34747f6`
- Git branch: `main`
- Python: `3.12.6`
- Seed policy: `not used; deterministic retrieval/eval path`
- Judge policy: `not used; deterministic metric implementation`

## Summary

| Metric | Score |
|---|---:|
| Retrieval recall@k | 0.8333 |
| Citation hit rate | 0.8333 |
| Refusal accuracy | 1.0000 |
| Faithfulness | 0.9000 |

## Examples

| ID | Status | Retrieval | Citation | Refusal | Faithfulness |
|---|---|---:|---:|---:|---:|
| q001 | answered | True | True |  | 1.00 |
| q002 | answered | True | True |  | 1.00 |
| q003 | answered | True | True |  | 1.00 |
| q004 | answered | True | True |  | 1.00 |
| q005 | answered | True | True |  | 1.00 |
| q006 | insufficient_context | False | False |  | 0.00 |
| q007 | answered | True | True |  | 1.00 |
| q008 | answered | True | True |  | 1.00 |
| q009 | answered | False | False |  | 0.50 |
| q010 | answered | False | False |  | 0.50 |
| q011 | answered | True | True |  | 1.00 |
| q012 | answered | True | True |  | 1.00 |
| q013 | answered | True | True |  | 1.00 |
| q014 | answered | True | True |  | 1.00 |
| q015 | answered | True | True |  | 1.00 |
| q016 | answered | True | True |  | 1.00 |
| q017 | answered | True | True |  | 1.00 |
| q018 | answered | True | True |  | 1.00 |
| q019 | insufficient_context | False | False | True | 1.00 |
| q020 | insufficient_context | False | False | True | 1.00 |
