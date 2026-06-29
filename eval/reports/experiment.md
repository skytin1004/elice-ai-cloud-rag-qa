# Part C Experiment Log

The same Gold Set and metric implementation were used for every run. The goal is not to hide failed attempts, but to observe how retrieval, citation, and refusal behavior change under controlled configuration changes.

## Summary

| Experiment | Strategy | Top-k | Min score | Rerank | Recall@k | Citation hit | Refusal | Faithfulness | Faithfulness delta |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| baseline-fixed-top3 | fixed | 3 | 0.00 | none | 0.8333 | 0.8333 | 1.0000 | 0.9000 | +0.0000 |
| heading-top5-threshold | heading | 5 | 0.08 | none | 0.8889 | 0.7778 | 1.0000 | 0.8500 | -0.0500 |
| heading-top8-relaxed | heading | 8 | 0.04 | none | 0.8889 | 0.7778 | 1.0000 | 0.8500 | -0.0500 |
| heading-top5-strict | heading | 5 | 0.16 | none | 0.8889 | 0.6111 | 1.0000 | 0.7250 | -0.1750 |
| heading-top8-keyword-rerank | heading | 8 | 0.04 | keyword | 0.8889 | 0.7778 | 1.0000 | 0.8500 | -0.0500 |

## Recovery Comparison

The initial heading-aware hypothesis failed against the fixed-size control. The recovery comparison treats that failed heading-aware setting as the new before state, then checks whether returning to a broad-context fixed chunking strategy restores the measured quality.

| Metric | Failed heading/top-5 | Broad fixed/top-3 | Delta |
|---|---:|---:|---:|
| Retrieval recall@k | 0.8889 | 0.8333 | -0.0556 |
| Citation hit rate | 0.7778 | 0.8333 | +0.0555 |
| Refusal accuracy | 1.0000 | 1.0000 | +0.0000 |
| Faithfulness | 0.8500 | 0.9000 | +0.0500 |

This is not a hidden replacement of the failed experiment. It is the follow-up design decision: preserve broad local context for answer generation, then handle stricter evidence selection as a separate next step.

## Per-Experiment Notes

### baseline-fixed-top3

- Hypothesis: Fixed-size chunks provide broad context and are a simple baseline.
- Result: recall=0.8333, citation=0.8333, refusal=1.0000, faithfulness=0.9000
- Analysis: This is the control run. It provides the comparison point for the later changes.
- Next step: Keep it as the control setting and compare every later change against it.

### heading-top5-threshold

- Hypothesis: Heading-aware chunks plus a threshold improve citation precision and refusal behavior.
- Result: recall=0.8889, citation=0.7778, refusal=1.0000, faithfulness=0.8500
- Analysis: The change did not improve the target metrics. A likely cause is that the fixed-size baseline retained broader local context, while the changed setting shifted the retrieved or cited source away from the expected evidence.
- Next step: Separate answer context selection from final citation selection.

### heading-top8-relaxed

- Hypothesis: Increasing top-k and relaxing the threshold can recover expected sources missed by narrower heading chunks.
- Result: recall=0.8889, citation=0.7778, refusal=1.0000, faithfulness=0.8500
- Analysis: The change did not improve the target metrics. A likely cause is that the fixed-size baseline retained broader local context, while the changed setting shifted the retrieved or cited source away from the expected evidence.
- Next step: Inspect missed examples before increasing context further; larger top-k alone may add noise.

### heading-top5-strict

- Hypothesis: A stricter threshold should reduce weak evidence but may lower recall.
- Result: recall=0.8889, citation=0.6111, refusal=1.0000, faithfulness=0.7250
- Analysis: The change did not improve the target metrics. A likely cause is that the fixed-size baseline retained broader local context, while the changed setting shifted the retrieved or cited source away from the expected evidence.
- Next step: Tune the threshold per query type instead of applying a single stricter global cutoff.

### heading-top8-keyword-rerank

- Hypothesis: Keyword-aware reranking can improve citation ordering after broad top-k retrieval.
- Result: recall=0.8889, citation=0.7778, refusal=1.0000, faithfulness=0.8500
- Analysis: The change did not improve the target metrics. A likely cause is that the fixed-size baseline retained broader local context, while the changed setting shifted the retrieved or cited source away from the expected evidence.
- Next step: Replace simple keyword overlap with BM25, heading proximity, and source diversity features.


## Analysis

The primary heading-aware hypothesis was not supported by this run. The fixed-size baseline remained stronger on at least one key metric, which suggests that wider chunks can be beneficial for this corpus/provider combination.

The highest faithfulness score in this sweep was `baseline-fixed-top3`. This does not automatically make it the production choice; implementation complexity, citation interpretability, and possible Gold Set overfitting still need to be considered.
