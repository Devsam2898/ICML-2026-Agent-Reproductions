# Claim 2: Zero-shot Accuracy


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_410f959130da", "created_at": "2026-07-24T07:14:28+00:00", "title": "Zero-shot accuracy on PIQA, WinoGrande, ARC-Easy, ARC-Challenge, OpenBookQA, an…"}
-->
Zero-shot accuracy on PIQA, WinoGrande, ARC-Easy, ARC-Challenge, OpenBookQA, and HellaSwag using the trained_weights.pt checkpoint from Claim 1. Target: accuracy close to the uncompressed baseline (minimal drop). Plan: eval.py --tasks piqa,winogrande,arc_easy,arc_challenge,openbookqa,hellaswag --batch-size 32.


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_ae1bda647153", "created_at": "2026-07-24T13:17:36+00:00", "title": "Baseline zero-shot numbers for Llama-3.1-8B-Instruct now available - see Claim…"}
-->
Baseline zero-shot numbers for Llama-3.1-8B-Instruct now available - see Claim 1 page for the full table and discrepancy writeup (PPL + zero-shot vs paper's Table 1 baseline row). Summary: avg zero-shot 68.49% ours vs 67.87% paper (+0.62pp, individual tasks vary more, HellaSwag +3.22pp is the largest outlier - likely lm_eval version drift). This baseline number is what our STAR-KV-compressed Llama-3.1-8B zero-shot results should be compared against once training completes.


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_8067ec8481b2", "created_at": "2026-07-26T06:12:33+00:00", "title": "Zero-shot results: STAR-KV compressed model (trainedweights.pt) (job https://hu…"}
-->
**Zero-shot results: STAR-KV compressed model (trained_weights.pt)** (job https://huggingface.co/jobs/Devavrat28/6a659ed5db23d7a7ec1cdc0c, starkv-eval-claim1, rtx-pro-6000, seed=42, tasks piqa/winogrande/arc_easy/arc_challenge/openbookqa/hellaswag, batch-size 32, `EXIT_EVAL=0`). lm_eval 0.4.12, n-shot=0 for all tasks.

acc_norm for OBQA/ARC-e/ARC-c/HellaSwag, acc for PIQA/WinoGrande (same convention used for our baseline, for consistency):
| Task | Paper baseline (0%) | Our baseline (0%) | Paper (60% comp) | Ours (60% comp) | Delta vs our baseline |
|---|---|---|---|---|---|
| OBQA | 43.00 | 43.00 | - | 41.60 | -1.40 |
| PIQA | 78.51 | 80.20 | - | 78.40 | -1.80 |
| ARC-e | 81.61 | 79.63 | - | 77.23 | -2.40 |
| ARC-c | 56.57 | 55.20 | - | 51.96 | -3.24 |
| HellaSwag | 75.95 | 79.17 | - | 74.47 | -4.70 |
| WinoGrande | 71.59 | 73.72 | - | 67.56 | -6.16 |
| **Avg** | **67.87** | **68.49** | **65.42** | **65.20** | **-3.29** |

(Paper only reports the aggregate 65.42% for the 60%-compression row in Table 1's summary we have on hand - no per-task compressed breakdown was available to compare against, so per-task deltas above are only vs. our own baseline.)

Comparison to paper's aggregate: ours 65.20% vs paper 65.42% (60% comp) - delta -0.22pp, very close. Our baseline-to-compressed drop is -3.29pp (68.49 -> 65.20), somewhat larger than the paper's own baseline-to-compressed drop of -2.45pp (67.87 -> 65.42), but the same direction and similar order of magnitude. WinoGrande shows the largest individual drop (-6.16pp), worth watching if we later probe per-task sensitivity in Extended Analysis.

Claim 2 status: VERIFIED (aggregate) - our compressed-model avg zero-shot accuracy (65.20%) matches the paper's reported 60%-compression aggregate (65.42%) within 0.22pp, well inside typical eval-run noise for a 6-task average.
