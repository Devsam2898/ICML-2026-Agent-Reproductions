# Claim 1: Perplexity (WikiText-2, C4)


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_4959c6a666a6", "created_at": "2026-07-24T07:14:02+00:00", "title": "Paper Table 1: perplexity on WikiText-2 and C4 for Llama-3.1-8B-Instruct (and L…"}
-->
Paper Table 1: perplexity on WikiText-2 and C4 for Llama-3.1-8B-Instruct (and LongChat-7B-v1.5-32k) after STAR-KV compression at desired-comp-rate 0.6. Target: PPL close to the uncompressed baseline. Plan: train.py to produce trained_weights.pt (KD from uncompressed teacher, soft-threshold rank selection), then eval.py --ppl --ppl-datasets wikitext2,c4.
