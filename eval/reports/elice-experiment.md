# Before/After Experiment

Hypothesis: heading-aware chunking plus a score threshold improves citation quality and refusal behavior compared with naive fixed-size chunking.

| Metric | Baseline fixed/top-3 | Improved heading/top-5 | Delta |
|---|---:|---:|---:|
| retrieval_recall_at_k | 1.0000 | 0.9444 | -0.0556 |
| citation_hit_rate | 1.0000 | 0.8889 | -0.1111 |
| refusal_accuracy | 1.0000 | 1.0000 | +0.0000 |
| faithfulness | 1.0000 | 0.9250 | -0.0750 |

Analysis: the hypothesis is not supported for this provider configuration. The fixed-size baseline outperformed the heading-aware setting on at least one key metric. A likely cause is that wider fixed chunks provided enough semantic context for the embedding model, while narrower heading chunks sometimes shifted the top-k context or citation away from the expected source.
