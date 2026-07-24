# EXPECTED_RESULTS.md: STAR-KV Ground Truth Numbers

## Purpose

This file contains every number from every table and key figure in the paper.
Compare your reproduction output against these values before declaring success or failure.

Acceptable margin: within 0.05 PPL and within 0.5% accuracy for zero-shot tasks,
unless otherwise noted. Latency numbers have higher variance (within 10% is acceptable).

Source: arXiv:2606.08382v1, Tables 1-5 and Appendix tables.

---

## Table 1: Perplexity and Zero-Shot Accuracy

### LongChat-7B-v1.5-32K

| Method | Comp% | Wiki2 PPL | C4 PPL | LM-avg PPL | OBQA | PIQA | ARC-e | ARC-c | Hella | Wino | Avg% |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Baseline | 0% | 6.86 | 9.93 | 8.40 | 41.00 | 76.28 | 71.84 | 41.38 | 71.20 | 67.48 | 61.53 |
| Palu | 50% | 7.37 | 11.67 | 9.52 | 38.20 | 73.78 | 67.63 | 37.63 | 68.43 | 65.27 | 58.49 |
| ReCalKV | 70% | 9.01 | 13.63 | 11.32 | 35.20 | 68.55 | 58.84 | 33.53 | 63.18 | 59.12 | 53.07 |
| **STAR-KV** | **60%** | **6.83** | **9.98** | **8.41** | **41.80** | **75.90** | **72.69** | **41.47** | **70.49** | **66.85** | **61.53** |
| **STAR-KV** | **75%** | **7.34** | **10.44** | **8.89** | **42.60** | **74.86** | **71.63** | **41.55** | **68.52** | **63.77** | **60.49** |

### LLaMA-2-7B

| Method | Comp% | Wiki2 PPL | C4 PPL | LM-avg PPL | OBQA | PIQA | ARC-e | ARC-c | Hella | Wino | Avg% |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Baseline | 0% | 5.12 | 7.04 | 6.08 | 44.20 | 78.07 | 76.30 | 46.42 | 76.00 | 69.30 | 65.05 |
| Palu | 50% | 5.63 | 8.38 | 7.01 | 43.60 | 76.33 | 73.02 | 42.57 | 73.39 | 66.67 | 62.60 |
| ReCalKV | 70% | 6.75 | 13.63 | 11.32 | 39.80 | 74.48 | 70.37 | 39.42 | 69.59 | 65.75 | 59.90 |
| **STAR-KV** | **60%** | **5.49** | **7.45** | **6.47** | **43.40** | **79.22** | **75.55** | **44.71** | **74.63** | **68.51** | **64.34** |
| **STAR-KV** | **75%** | **5.86** | **8.02** | **6.94** | **43.00** | **78.18** | **73.78** | **41.64** | **73.14** | **66.61** | **62.73** |

### LLaMA-3-8B-Instruct

| Method | Comp% | Wiki2 PPL | C4 PPL | LM-avg PPL | OBQA | PIQA | ARC-e | ARC-c | Hella | Wino | Avg% |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Baseline | 0% | 7.74 | 12.61 | 10.18 | 43.00 | 78.51 | 81.61 | 56.57 | 75.95 | 71.59 | 67.87 |
| Palu | 50% | 9.43 | 17.37 | 13.40 | 42.40 | 76.12 | 76.14 | 49.06 | 70.33 | 71.82 | 64.31 |
| **STAR-KV** | **60%** | **8.52** | **13.51** | **11.01** | **42.20** | **78.13** | **79.29** | **49.83** | **73.37** | **69.69** | **65.42** |
| **STAR-KV** | **75%** | **10.20** | **15.46** | **12.83** | **42.20** | **78.02** | **77.57** | **49.32** | **71.59** | **66.22** | **64.15** |

---

## Table 2: LongBench Evaluation

### LongChat-7B-v1.5-32K

| Method | Comp% | Qasper | QMSum | TriviaQA | MultiQA | TREC | MultiNews | VCSum | Avg |
|---|---|---|---|---|---|---|---|---|---|
| Baseline | 0% | 27.7 | 22.7 | 82.3 | 43.0 | 68.5 | 26.1 | 15.4 | 40.81 |
| Palu | 30% | 23.81 | 22.64 | 80.61 | 44.15 | 64.5 | 25.4 | 14.08 | 39.31 |
| Palu | 50% | 21.1 | 22.4 | 75.81 | 40.78 | 62.5 | 22.6 | 12.58 | 36.80 |
| **STAR-KV** | **60%** | **22.58** | **22.4** | **81.3** | **41.04** | **65.0** | **25.7** | **15.6** | **39.09** |

### LLaMA-3.1-8B-Instruct

| Method | Comp% | Qasper | QMSum | TriviaQA | MultiQA | TREC | MultiNews | VCSum | Avg |
|---|---|---|---|---|---|---|---|---|---|
| Baseline | 0% | 25.19 | 23.2 | 92.00 | 39.90 | 72.5 | 26.9 | 15.91 | 42.23 |
| Palu | 30% | 14.47 | 23.3 | 86.71 | 26.57 | 73.0 | 26.19 | 8.33 | 36.93 |
| Palu | 50% | 15.38 | 22.10 | 73.36 | 27.58 | 63.5 | 21.66 | 1.95 | 32.21 |
| **STAR-KV** | **60%** | **23.2** | **22.4** | **89.06** | **36.34** | **67.0** | **25.89** | **13.2** | **39.58** |
| **STAR-KV** | **50%** | **23.15** | **22.57** | **89.32** | **38.29** | **71.50** | **26.29** | **14.43** | **40.79** |

---

## Table 3: RULER Evaluation at 4K Sequence Length

### LongChat-7B-v1.5-32K

| Method | Comp% | MK1 | MK2 | MQ | MV | S1 | S2 | S3 | FWE | SQ | Avg |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Baseline | 0% | 99.80 | 99.60 | 98.40 | 96.55 | 100.0 | 100.0 | 100.0 | 48.20 | 56.72 | 88.80 |
| Palu | 50% | 99.80 | 98.80 | 75.30 | 97.40 | 100.0 | 100.0 | 98.60 | 42.40 | 53.25 | 85.06 |
| **STAR-KV** | **60%** | **99.40** | **99.60** | **98.50** | **96.55** | **100.0** | **100.0** | **99.80** | **46.00** | **54.45** | **88.26** |
| STAR-KV diff. seed | 60% | 99.20 | 99.80 | 98.35 | 97.60 | 100.0 | 100.0 | 100.0 | 49.93 | 53.48 | 88.71 |

### LLaMA-3.1-8B-Instruct

| Method | Comp% | MK1 | MK2 | MQ | MV | S1 | S2 | S3 | FWE | SQ | Avg |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Baseline | 0% | 100.0 | 99.8 | 99.9 | 98.95 | 100.0 | 100.0 | 99.6 | 96.07 | 78.12 | 96.94 |
| Palu | 50% | 98.60 | 99.80 | 75.35 | 69.65 | 99.80 | 96.60 | 88.40 | 85.13 | 58.58 | 85.06 |
| **STAR-KV** | **60%** | **98.4** | **86.0** | **85.1** | **84.0** | **100.0** | **100.0** | **92.0** | **87.73** | **68.05** | **89.03** |

---

## Table 4: Quantization Results (LongChat-7B-v1.5-32K)

| Method | Avg bit | Comp rate | Avg zero-shot % |
|---|---|---|---|
| Baseline | 16 | 1x | 60.34 |
| KIVI-2-gs32-r32 | 3.16 | 5.1x | 60.3 |
| KVQuant-2bit-1% | 2.33 | 6.9x | 58.00 |
| Palu (50%) | 3 | 10.4x | 56.42 |
| Palu (50%) | 4 | 8.0x | 56.71 |
| **STAR-KV (60%)** | **3.2** | **12.5x** | **60.12** |
| **STAR-KV (75%)** | **3.2** | **20.0x** | **59.18** |

---

## Table 5: End-to-End Generation Throughput (LLaMA-2-7B, RTX 4090)

| Context length | PyTorch SDPA tokens/s (batch size) | STAR-KV tokens/s (batch size) | Gain |
|---|---|---|---|
| 1K | 249.8 (16) | 751.0 (128) | 3.01x |
| 2K | 130.4 (8) | 400.7 (64) | 3.07x |
| 4K | 69.0 (4) | 212.2 (32) | 3.07x |
| 8K | 35.6 (2) | 110.6 (16) | 3.11x |
| 16K | 18.0 (1) | 56.6 (8) | 3.14x |

Average gain: 3.1x

---

## Attention Speedup (Figure 9, LLaMA-2-7B, batch size 16, RTX 4090)

Key values to verify:

| Context | STAR-KV 75% (no quant) | STAR-KV 75% + 4-bit |
|---|---|---|
| 8K | ~2.6x | ~2.2x |
| 16K | ~3.3x | ~3.5x |
| 32K | ~4.1x (vs estimated baseline) | ~4.9x (vs estimated baseline) |
| 64K | N/A (OOM no quant) | ~6.9x (vs estimated baseline) |

NOTE: 32K and 64K baselines are ESTIMATED by the authors (PyTorch OOM on RTX 4090).
Your reproduction should note this when reporting these numbers.

---

## RULER at 16K (Appendix Table 13, LongChat-7B-v1.5-32K)

| Method | Comp% | RULER@16K Avg |
|---|---|---|
| Baseline | 0% | 85.69 |
| Palu | 60% | 62.94 |
| **STAR-KV** | **60%** | **84.16** |

---

## Appendix Table 6: Static vs Adaptive Rank Selection

| Method | Comp% | ARC-e |
|---|---|---|
| Static Low-rank | 75% | 37.52 |
| **STAR-KV adaptive** | **75%** | **71.63** |

This is a key ablation. The 34 percentage point gap shows why adaptive rank matters.

---

## Appendix A.1.3: Calibration Dataset Sensitivity (LongChat-7B-v1.5-32K, 60%)

| Calibration data | Avg zero-shot % |
|---|---|
| Baseline | 61.53 |
| FineWeb-Edu | 61.53 |
| C4 | 61.39 |
| RedPajama | 61.18 |
| FineWeb-Edu (diff. seed) | 61.15 |

Acceptable range: within 0.68% across datasets per paper findings.

---

## Appendix A.1.5: Compression Weight Sensitivity (LongChat-7B-v1.5-32K, 60%)

| gamma | Avg zero-shot % |
|---|---|
| 1.0 | ~61.28 (see paper for full row) |
| 0.1 | 61.53 (used in main experiments) |
| 0.01 | 61.33 |

The paper uses gamma = 0.1 as default. Variation is within 0.25%.

---

## Appendix Table 12: Additional Models

### Mistral-7B-Instruct-v0.2

| Method | Comp% | Wiki2 | C4 | Avg% |
|---|---|---|---|---|
| Baseline | 0% | 5.94 | 9.72 | 70.33 |
| Palu | 60% | 7.07 | 12.93 | 65.03 |
| ReCalKV | 60% | N/A | N/A | 67.24 |
| **STAR-KV** | **60%** | **8.01** | **10.81** | **67.73** |

### LLaMA-2-13B

| Method | Comp% | Wiki2 | C4 | Avg% |
|---|---|---|---|---|
| Baseline | 0% | 5.71 | 8.19 | 67.04 |
| Palu | 50% | 6.48 | 9.69 | 62.45 |
| **STAR-KV** | **60%** | **5.69** | **8.22** | **66.44** |
| **STAR-KV** | **75%** | **5.89** | **8.66** | **64.75** |

---

## Priority Order for Reproduction

Reproduce in this order. Stop and document any mismatch before continuing.

1. Table 1 baseline rows (uncompressed model)
2. Table 1 STAR-KV at 60% for LongChat-7B
3. Table 1 STAR-KV at 60% for LLaMA-3.1-8B
4. Table 4 quantization result (20x compression)
5. Table 5 throughput numbers
6. Table 2 LongBench
7. Table 3 RULER at 4K
8. Appendix ablations (only if time and budget allow)
