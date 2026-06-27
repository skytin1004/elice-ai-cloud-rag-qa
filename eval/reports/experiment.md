# Before/After Experiment

Hypothesis: heading-aware chunking plus a score threshold improves citation quality and refusal behavior compared with naive fixed-size chunking.

| Metric | Baseline fixed/top-3 | Improved heading/top-5 | Delta |
|---|---:|---:|---:|
| retrieval_recall_at_k | 0.8333 | 0.8889 | +0.0556 |
| citation_hit_rate | 0.8333 | 0.7778 | -0.0555 |
| refusal_accuracy | 1.0000 | 1.0000 | +0.0000 |
| faithfulness | 0.9000 | 0.8500 | -0.0500 |

Analysis: this report is generated automatically from the same gold set and provider configuration. Any metric regression should be analyzed by inspecting the per-example baseline and improved reports next to this comparison.
