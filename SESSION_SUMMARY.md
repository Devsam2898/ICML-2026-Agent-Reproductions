# Session Summary — 2026-07-24

## What we did

- Cloned `STAR-KV` (official repo) and `Beyond-Perplexity-TurboQuant` (our prior work)
  as git submodules; created `repro/`, `results/`; wrote a logbook-style `README.md`.
- Pushed everything to https://github.com/Devsam2898/ICML-2026-Agent-Reproductions.
- Installed Trackio, logged into HF (`Devavrat28`, fine-grained token with write +
  `job.write` scopes), and published a per-claim Trackio logbook:
  **https://huggingface.co/spaces/Devavrat28/star-kv** — this has the full detailed
  writeups; this file is just a pointer/summary.
- Ran Triton kernel correctness checks (`abx_rope_batched.py --check`,
  `bx_quant.py --check`) on HF Jobs — **both fail** (device-placement bug in the
  reference path; `quant_utils.py` is missing from the repo entirely). Blocks Claim 4
  and the quantized half of Claim 3.
- User provided `SKILLS.md` / `EXPECTED_RESULTS.md` / `EXTENDED_ANALYSIS.md` — the
  authoritative paper knowledge base, ground-truth numbers, and reproduction protocol
  ("paper is ground truth; document discrepancies before any workaround, never
  silently fix"). This corrected the plan:
  - Paper's actual hardware is **RTX PRO 6000 (96GB)**, not A100 — explains an
    earlier training OOM on a 80GB A100.
  - A **baseline (uncompressed) eval step** was missing from the original plan and
    is required before trusting any compressed-model numbers.
  - LongChat-7B-v1.5-32k takes priority over Llama-3.1-8B for Table 1 STAR-KV repro.
- Ran the baseline eval (rtx-pro-6000): **Llama-3.1-8B-Instruct** succeeded —
  Wiki2 PPL 7.21 (paper 7.74), C4 PPL 11.40 (paper 12.61), avg zero-shot 68.49%
  (paper 67.87%, HellaSwag the biggest single-task outlier at +3.22pp). Mismatch is
  outside tolerance but user decided to document-and-continue (likely `lm_eval`/
  `transformers`/`datasets` version drift, not a real bug).
  **LongChat-7B-v1.5-32k** baseline is still blocked (tokenizer loading fails under
  the installed `transformers` version even after adding `tiktoken`).
- All discrepancies are logged on the logbook's **Discrepancies Log** page with the
  required structure (paper says / code does / likely cause / proposed fix / impact),
  per SKILLS.md protocol. Four logged so far; two have workarounds applied
  (`repro/compat_shim.py`, added `tiktoken`), two are open (baseline PPL mismatch,
  LongChat tokenizer failure).
- Submitted the STAR-KV training job (Llama-3.1-8B, 60% compression) on the
  corrected rtx-pro-6000 hardware, using `repro/seeded_run.py` for determinism
  (train.py has no seeding of its own — also documented as a discrepancy).

## In flight right now

- **Job `6a6366addb23d7a7ec1ca8ba`**, rtx-pro-6000, submitted 2026-07-24 13:20 UTC,
  4h timeout cap (~$11 max, likely 1-3h/~$3-8 in practice per CLAUDE.md's estimate).
  Check status: `hf jobs inspect Devavrat28/6a6366addb23d7a7ec1ca8ba`.
  It runs independently of this session — no need to keep a terminal open.
  Output (`trained_weights.pt`, `fused_weights.pt`) lands in the
  `hf://buckets/Devavrat28/star-kv-checkpoints` bucket.

## What's next

1. Check the training job result; log it (PPL/zero-shot loss curve, exit code,
   wall clock, peak GPU memory) to the Claim 1 logbook page either way.
2. Run `eval.py` (real, compressed-model path) on the resulting checkpoint for
   PPL (WikiText-2/C4) and zero-shot accuracy; compare against paper's 60%-compression
   row (Wiki2 8.52, C4 13.51, avg zero-shot 65.42%) **and** our own baseline.
3. Fix LongChat's tokenizer loading (try `use_fast=False` first — cheap, one job),
   then run its baseline eval, then its STAR-KV training (priority order puts
   LongChat @ 60% ahead of Llama per `EXPECTED_RESULTS.md`).
4. Claim 3 (compression ratio): low-rank-only number derivable from the trained
   checkpoint; the 20x combined/quantized number is blocked on missing `quant_utils.py`.
5. Claim 4 (speedup/throughput): blocked until the two failing kernel checks are
   understood/resolved (not by patching `STAR-KV/` — investigate root cause first).
6. Claim 5 (LongBench/RULER): not started.
7. Extended Analysis (Qwen2.5-3B + TurboQuant's 5 geometric metrics): not started,
   and per `EXTENDED_ANALYSIS.md` should only begin once the official reproduction
   (Tables 1 and 4) is verified.

## Key references

- Logbook (full detail): https://huggingface.co/spaces/Devavrat28/star-kv
- Repo: https://github.com/Devsam2898/ICML-2026-Agent-Reproductions
- `SKILLS.md` / `EXPECTED_RESULTS.md` / `EXTENDED_ANALYSIS.md` at repo root —
  read these before resuming; they're the ground truth for what "correct" looks like.
