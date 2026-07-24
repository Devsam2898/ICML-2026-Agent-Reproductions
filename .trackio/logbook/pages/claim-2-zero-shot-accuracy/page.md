# Claim 2: Zero-shot Accuracy


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_410f959130da", "created_at": "2026-07-24T07:14:28+00:00", "title": "Zero-shot accuracy on PIQA, WinoGrande, ARC-Easy, ARC-Challenge, OpenBookQA, an…"}
-->
Zero-shot accuracy on PIQA, WinoGrande, ARC-Easy, ARC-Challenge, OpenBookQA, and HellaSwag using the trained_weights.pt checkpoint from Claim 1. Target: accuracy close to the uncompressed baseline (minimal drop). Plan: eval.py --tasks piqa,winogrande,arc_easy,arc_challenge,openbookqa,hellaswag --batch-size 32.
