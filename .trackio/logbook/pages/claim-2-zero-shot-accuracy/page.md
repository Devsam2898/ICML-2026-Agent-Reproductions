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
