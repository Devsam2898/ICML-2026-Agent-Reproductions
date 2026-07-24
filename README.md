# STAR-KV Reproduction Logbook

Running notes for the ICML 2026 Reproducibility Challenge submission covering
**STAR-KV: Low-Rank KV Cache Compression via Soft Thresholding for Adaptive Rank Control**
(arXiv:2606.08382, https://github.com/PriyanshBhatnagar/STAR-KV).

Challenge: https://huggingface.co/spaces/ICML-2026-agent-repro/challenge

See `CLAUDE.md` for the original plan/commands, and `SKILLS.md` / `EXPECTED_RESULTS.md` /
`EXTENDED_ANALYSIS.md` for the authoritative paper knowledge base, ground-truth numbers,
and reproduction protocol (paper is ground truth; discrepancies are documented before any
workaround, never silently fixed - see the Discrepancies Log page in the Trackio logbook).

---

## Status

| Step | Status |
|---|---|
| Clone official repo into `STAR-KV/` (git submodule) | Done |
| Add prior TurboQuant work as `Beyond-Perplexity-TurboQuant/` (git submodule) | Done |
| Conda env `StarKV` (python 3.12.7) + requirements.txt | N/A — using HF Jobs GPU instead (see below) |
| HF_TOKEN / WANDB_API_KEY configured | Done (fine-grained token, write + job.write scopes) |
| Trackio logbook published | Done — https://huggingface.co/spaces/Devavrat28/star-kv |
| GPU compute path | HF Jobs, `rtx-pro-6000` flavor (matches paper's actual hardware per SKILLS.md; A100 OOM'd on training - see below) |
| Triton kernel smoke tests (`abx_rope_batched.py --check`, `bx_quant.py --check`) | **Both fail** — see below |
| Baseline eval (uncompressed) — Llama-3.1-8B-Instruct | Done — PPL/zero-shot mismatch vs paper, documented |
| Baseline eval (uncompressed) — LongChat-7B-v1.5-32k | Blocked — tokenizer load fails (transformers version issue) |
| Claim 1: PPL (WikiText-2, C4) | Baseline done; STAR-KV training not yet retried on correct hardware |
| Claim 2: Zero-shot accuracy | Baseline done; compressed-model run pending |
| Claim 3: KV compression ratio | At risk — quantized path broken (see below) |
| Claim 4: Attention speedup / e2e throughput | Blocked — kernel checks fail |
| Claim 5: LongBench / RULER | Pending |
| Extended analysis (Qwen2.5-3B + TurboQuant metrics) | Pending |

---

## Paper Claims Summary Table

Ground-truth values from `EXPECTED_RESULTS.md` (Table 1, Llama-3.1-8B-Instruct).

| Claim | Paper Value | Our Result | Status |
|---|---|---|---|
| Baseline PPL WikiText-2 (Llama-3.1-8B, 0% comp) | 7.74 | 7.21 | Mismatch (-0.53, outside 0.05 tolerance) |
| Baseline PPL C4 (Llama-3.1-8B, 0% comp) | 12.61 | 11.40 | Mismatch (-1.21, outside 0.05 tolerance) |
| Baseline avg zero-shot (Llama-3.1-8B, 0% comp) | 67.87% | 68.49% | Close (+0.62pp avg; HellaSwag +3.22pp outlier) |
| PPL WikiText-2 (Llama-3.1-8B, 60% comp) | 8.52 | TBD | Pending (training not yet retried) |
| PPL C4 (Llama-3.1-8B, 60% comp) | 13.51 | TBD | Pending |
| KV compression ratio (low-rank only) | up to 75% | TBD | Pending |
| KV compression ratio (combined, quantized) | up to 20x | Blocked | quant_utils.py missing |
| Attention speedup (Triton) | up to 6.9x | Blocked | kernel `--check` fails |
| E2E throughput improvement | up to 3.1x | Blocked | kernel `--check` fails |
| Zero-shot avg accuracy drop (60% comp) | ~minimal (65.42% reported) | TBD | Pending |

---

## Log Entries

Each entry should record: command run, GPU used and runtime, exact output numbers,
comparison to the paper's reported numbers, and notes on any deviation or failure.
Corresponding raw JSON output belongs in `results/`. Full detail lives in the
[published Trackio logbook](https://huggingface.co/spaces/Devavrat28/star-kv); this
section is a condensed pointer.

### 2026-07-24 — Environment setup

- No local GPU/conda; compute runs via `hf jobs run` on HF Jobs (A100, credit from
  the ICML-2026-agent-repro org). Trackio logbook opened and published.

### 2026-07-24 — Triton kernel correctness checks (Claim 4 pre-flight)

- Job: `hf jobs run --flavor a100-large python:3.12 bash repro/kernel_check.sh`
  ([job](https://huggingface.co/jobs/Devavrat28/6a633a27db23d7a7ec1ca3c3), 80s runtime, ~$0.06)
- `abx_rope_batched.py --check`: **FAILS** —
  `RuntimeError: Expected all tensors to be on the same device, but got mat2 is on
  cpu, different from other tensors on cuda:0`, raised inside the `torch_abx()`
  reference path at `rotary_emb(xb, position_ids)`.
- `bx_quant.py --check`: **FAILS** — `ModuleNotFoundError: No module named
  'quant_utils'`. `quant_utils.py` does not exist anywhere in the cloned repo
  (confirmed via full-tree grep before running), yet both `bx_quant.py` and
  `LlamaLoRaAttention_headwise_quant.py` import from it.
- Confirms the upstream repo's own TODO ("Fix kernels for acc analysis") is real.
  Not modifying `STAR-KV/` to patch this per project convention — logged as a
  reproduction finding instead. Claim 4 (speedup/throughput) is blocked until
  upstream fixes land; Claim 3's combined (low-rank + quantization) ratio is at
  risk since the quantized attention path can't import.

### 2026-07-24 — SKILLS.md / EXPECTED_RESULTS.md / EXTENDED_ANALYSIS.md added

- User-provided ground-truth docs corrected several things: paper's actual hardware is
  RTX PRO 6000 (96GB), not A100; a baseline (uncompressed) eval step is required before
  training, which the original plan skipped; LongChat-7B-v1.5-32k takes priority over
  Llama-3.1-8B for Table 1 STAR-KV reproduction. Adopted going forward.

### 2026-07-24 — Training attempt #1: CUDA OOM on A100

- Job [6a6347b4db23d7a7ec1ca567](https://huggingface.co/jobs/Devavrat28/6a6347b4db23d7a7ec1ca567),
  a100-large (1x80GB), 142s, ~$0.06.
- Teacher+student (both Llama-3.1-8B, bf16) at seq-len=8192 exceed 80GB;
  `train.py` has no gradient checkpointing. Confirmed by SKILLS.md: authors used
  RTX PRO 6000 (96GB) for this exact step. Will retry there.

### 2026-07-24 — Baseline (uncompressed) eval

- Job [6a636385db23d7a7ec1ca87f](https://huggingface.co/jobs/Devavrat28/6a636385db23d7a7ec1ca87f),
  rtx-pro-6000, seed=42, bf16, ~7.5 min total, ~$0.34.
- **Llama-3.1-8B-Instruct**: Wiki2 PPL 7.21 (paper 7.74), C4 PPL 11.40 (paper 12.61),
  avg zero-shot 68.49% (paper 67.87%). Individual zero-shot tasks vary more than the
  0.5% tolerance (HellaSwag +3.22pp); OBQA matches paper exactly (43.00). Documented
  as a discrepancy (likely `lm_eval`/`transformers`/`datasets` version drift, since
  STAR-KV's `requirements.txt` pins none of these) - see Discrepancies Log.
- **LongChat-7B-v1.5-32k**: still blocked. First attempt failed on missing `tiktoken`;
  after adding it, fails differently (`transformers` misroutes its sentencepiece
  tokenizer through tiktoken's BPE parser). Next attempt: `use_fast=False`.
- Two environment fixes applied without touching `STAR-KV/`: added `tiktoken`/
  `sentencepiece` to the install list, and `repro/compat_shim.py` monkeypatches
  `datasets.load_dataset` to redirect the legacy bare `wikitext` id (which newer
  `datasets` rejects) to `Salesforce/wikitext`.

---

## Extended Analysis Dependency

`Beyond-Perplexity-TurboQuant/` (git submodule) holds the prior TurboQuant work
referenced by CLAUDE.md's Extended Analysis section — `experiment_1.py` implements
the 5 geometric metrics (attention KL divergence, per-layer sensitivity, norm vs
direction error, token-position degradation, GQA cross-query consistency) to be run
against STAR-KV's compressed Qwen2.5-3B outputs, to compare against the Layer 0
key-norm heterogeneity finding from that project.

## Notes / Known Issues

- Repo TODOs at clone time (from `STAR-KV/README.md`): missing trained weights for
  LongChat/LLaMA-3.1-8B, kernels not yet fixed for accuracy analysis, quantization
  latency tests not yet added.
- This machine has no `conda` on PATH and no local GPU verified yet — training/eval
  needs to happen wherever the HuggingFace GPU credit is provisioned (e.g. an HF Space
  or a GPU box with `HF_TOKEN`/`WANDB_API_KEY` set).
- `myenv/` is a pre-existing local Python venv unrelated to the `StarKV` conda env
  specified by the paper's install instructions.
