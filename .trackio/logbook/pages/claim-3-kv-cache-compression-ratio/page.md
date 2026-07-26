# Claim 3: KV Cache Compression Ratio


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_890eaddee25b", "created_at": "2026-07-24T07:14:39+00:00", "title": "KV cache compression ratio at desired-comp-rate 0.6: paper claims up to 75% red…"}
-->
KV cache compression ratio at desired-comp-rate 0.6: paper claims up to 75% reduction from low-rank projection alone, and up to 20x combined with the low-rank-aware mixed-precision quantization. Plan: derive achieved rank/compression from the trained soft-threshold checkpoint and compare against the paper's reported ratios.


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_12b2c21dd1c2", "created_at": "2026-07-24T09:21:14+00:00", "title": "Note: 'quantutils.py' (required by bxquant.py / LlamaLoRaAttentionheadwisequant…"}
-->
Note: 'quant_utils.py' (required by bx_quant.py / LlamaLoRaAttention_headwise_quant.py for the int8/int4 KV quantization path) is missing from the cloned repo as of this session - see Claim 4 page for details. The up-to-20x combined (low-rank + quantization) ratio may not be reproducible until that file is recovered or reconstructed; the up-to-75% low-rank-only ratio is unaffected.


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_d90ca4332c5a", "created_at": "2026-07-26T06:23:53+00:00", "title": "Achieved low-rank-only compression ratio (from training job https://huggingface…"}
-->
**Achieved low-rank-only compression ratio** (from training job https://huggingface.co/jobs/Devavrat28/6a6505dadb23d7a7ec1cced8, `--desired-comp-rate 0.6`, `--comp-weight-k 0.1 --comp-weight-v 0.1`, `--alpha-samples 3000`):

| Component | Target (per-component, from desired-comp-rate 0.6) | Achieved | Notes |
|---|---|---|---|
| K (headwise) | min(0.6+0.09, 0.99) = 0.69 | 0.6195 | Phase 1 ended via alpha-samples=3000 step-count fallback, not budget-reached (see training-success cell on Claim 1 page and Discrepancies Log) |
| V (joint) | max(0.6-0.11, 0.01) = 0.49 | 0.4933 | Reached/frozen at step 392 ("[V frozen]" log line) - budget reached, not step-count fallback |
| Average | 0.60 | ~0.5564 | (0.6195+0.4933)/2 |

Paper claims "up to 75%" reduction from low-rank alone - our run targeted a lower 60% average operating point (matching the paper's own Table 1 60%-compression row used for the PPL/zero-shot comparisons), so 75% is the paper's ceiling across compression settings, not the specific number we should hit at this operating point. At the 0.6 operating point, our achieved average (~55.6%) is below the 60% target, consistent with the K-component undershoot noted above (V hit its target/froze; K did not fully converge within the fixed alpha-samples budget).

Combined (low-rank + int8/int4 quantization) ratio, claimed up to 20x: **not reproducible** - `quant_utils.py` (required by `bx_quant.py` / the quantized attention path) is missing from the cloned repo (see Claim 4 page and Discrepancies Log for the same finding). This blocks only the *combined* ratio; the low-rank-only ratio above is unaffected by this gap.

Claim 3 status: PARTIALLY VERIFIED - low-rank-only compression is real and roughly in the right range (~56% average vs 60% target, V exactly on target, K short by ~7pp due to a step-budget fallback rather than a methodology failure), but we have not reproduced the paper's specific "up to 75%" ceiling claim (would need a run targeting a higher desired-comp-rate to test that specifically) and cannot reproduce the "up to 20x combined" claim at all due to the missing quantization module.
