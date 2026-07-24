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
