# Discrepancies Log


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_65c9470f6807", "created_at": "2026-07-24T12:48:59+00:00", "title": "Central log of paper-vs-code discrepancies, per SKILLS.md protocol (never silen…"}
-->
Central log of paper-vs-code discrepancies, per SKILLS.md protocol (never silently fix; document before touching code).


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_c41ab5b4b67f", "created_at": "2026-07-24T12:51:03+00:00", "title": "DISCREPANCY FOUND"}
-->
**DISCREPANCY FOUND**
Location: STAR-KV/train.py (entire file)
Paper says (SKILLS.md determinism requirement): every experiment must fix all sources of randomness (torch/numpy/random seeds, cudnn deterministic mode) so runs are reproducible.
Code does: train.py has zero seed-setting anywhere - no random.seed/np.random.seed/torch.manual_seed/torch.cuda.manual_seed_all calls, no cudnn.deterministic flag.
Difference: as shipped, two runs of train.py with identical args are not guaranteed reproducible (weight init randomness is fixed by from_pretrained, but dataloader shuffling/sampling and any dropout are not seeded).
Likely cause: Documentation bug / implementation gap - determinism was presumably handled by the authors' own run harness, not included in the public release.
Proposed fix: do not modify STAR-KV/train.py. Instead use a wrapper (repro/seeded_run.py) that sets all seeds via runpy before executing train.py's __main__, so determinism is added without touching the vendored file.
Impact on results: low - affects exact bit-for-bit reproducibility of a given run, not expected to shift metrics outside normal run-to-run noise. Documented per protocol regardless.


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_11ca112743d2", "created_at": "2026-07-24T12:52:27+00:00", "title": "DISCREPANCY FOUND"}
-->
**DISCREPANCY FOUND**
Location: STAR-KV/eval.py lines 145-217 (parse_args --weights required=True; main() unconditionally calls replace_linear_layer then model.load_state_dict(torch.load(args.weights)))
Paper says (SKILLS.md Step 3 / EXPECTED_RESULTS.md): must evaluate the uncompressed baseline model on WikiText-2/C4/zero-shot BEFORE training, and compare against the paper's baseline rows (e.g. LongChat baseline Wiki2 PPL 6.86; Llama-3.1-8B-Instruct baseline Wiki2 PPL 7.74).
Code does: eval.py's main() always injects STAR-KV's low-rank decomposed linear layers via replace_linear_layer() and always requires a --weights checkpoint to be loaded via model.load_state_dict() before any evaluation runs. There is no argument or code path to evaluate the stock/uncompressed HF checkpoint through eval.py's CLI.
Difference: the shipped eval.py cannot produce a 'Comp%=0%' baseline row; baseline numbers must have been produced by the authors outside this released script.
Likely cause: Implementation bug / missing feature (baseline eval tooling not included in public release).
Proposed fix: do not modify STAR-KV/eval.py. Instead, reuse its evaluate_ppl(model, tokenizer, datasets, seqlen, device) and evaluate_lmeval(model, tokenizer, tasks, batch_size, ...) functions directly - both take a model object as a parameter and don't require going through main()/--weights. Wrapper: repro/baseline_eval.py imports these from eval.py as a library, loads the stock LlamaForCausalLM.from_pretrained(model_name) with no decomposition applied, and calls them directly.
Impact on results: none on compressed-model numbers; without this wrapper we would be unable to independently verify the paper's own reported baseline rows and could only compare our compressed results against the paper's baseline as given, not our own reproduction of it.


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_b01ead067e4d", "created_at": "2026-07-24T13:06:47+00:00", "title": "Environment/dependency issues found on first baseline-eval job run (job https:/…"}
-->
**Environment/dependency issues found on first baseline-eval job run** (job https://huggingface.co/jobs/Devavrat28/6a6361c87ef3c0846496779c, rtx-pro-6000, ~125s, ~$0.06):

1. lmsys/longchat-7b-v1.5-32k tokenizer load failed: ModuleNotFoundError: No module named 'tiktoken' (then ValueError requiring tiktoken to read a tiktoken file). requirements.txt does not list tiktoken as a transitive dependency of this tokenizer. Fix: add tiktoken (and sentencepiece, for parity) to the install list in our job scripts - not a change to STAR-KV/ itself, just our own environment setup script.

2. Llama-3.1-8B baseline model loaded fine, but evaluate_ppl -> _get_ppl_data('wikitext2', ...) -> load_dataset('wikitext', 'wikitext-2-raw-v1') raised huggingface_hub.errors.HfUriError: 'Repository id must be namespace/name, got wikitext'. The unpinned latest 'datasets' package (5.0.0 in this environment) no longer resolves the bare legacy dataset id STAR-KV/eval.py hardcodes at line 56. This is the 'Version mismatch' failure mode SKILLS.md explicitly warns about. Per protocol, not modifying STAR-KV/eval.py. Fix: repro/compat_shim.py monkeypatches datasets.load_dataset to redirect 'wikitext' -> 'Salesforce/wikitext' (falling back to the original id if that alias fails), imported before eval.py/train.py in both repro/baseline_eval.py and repro/seeded_run.py so it also covers the real (non-baseline) eval.py runs later.


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_0f7aac0d452b", "created_at": "2026-07-25T17:00:10+00:00", "title": "DISCREPANCY FOUND"}
-->
**DISCREPANCY FOUND**
Location: repro/seeded_run.py (our own wrapper, not STAR-KV/)
Paper says: n/a - bug in our reproduction tooling, not the paper or repo.
Code does: seeded_run.py called runpy.run_path(target, run_name="__main__") without adding the target script's own directory to sys.path first.
Difference: python train.py (direct invocation) implicitly prepends train.py's directory to sys.path, letting its sibling import (from model import ...) resolve. runpy.run_path does not do this automatically - confirmed empirically with a minimal repro. Training job 6a6366addb23d7a7ec1ca8ba failed after 75s with ModuleNotFoundError: No module named 'model', before any real training occurred.
Likely cause: our own tooling bug (wrapper gap), not an upstream STAR-KV issue.
Proposed fix: applied - seeded_run.py now does sys.path.insert(0, os.path.dirname(os.path.abspath(target))) before runpy.run_path. Verified fix against a minimal local reproduction before resubmitting.
Impact on results: none on paper claims - only affected our determinism wrapper, not STAR-KV/train.py itself. Job resubmitted as 6a64eb397ef3c08464968896 (starkv-train-claim1-v3) with the fix.
