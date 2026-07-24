# Extended Analysis: Qwen2.5-3B + TurboQuant Metrics


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_5855c2fca719", "created_at": "2026-07-24T07:22:03+00:00", "title": "Our original contribution beyond the required reproduction (see CLAUDE.md Exten…"}
-->
Our original contribution beyond the required reproduction (see CLAUDE.md Extended Analysis). Prior TurboQuant work (github.com/Devsam2898/Beyond-Perplexity-TurboQuant) found Qwen2.5-3B-Instruct is unstable under KV compression due to Layer 0 key-norm heterogeneity (K/V norm ratio ~52x). Plan: (1) apply STAR-KV's soft-threshold low-rank compression to Qwen2.5-3B-Instruct's attention layers, (2) run TurboQuant's experiment_1.py 5 geometric metrics on top of it: M1 attention KL divergence, M2 per-layer sensitivity curve, M3 norm vs direction error, M4 token-position degradation, M5 GQA cross-query consistency. Question: does STAR-KV's low-rank approach resolve or reproduce the Layer 0 anomaly found with TurboQuant's quantization approach?
