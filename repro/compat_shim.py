"""Environment-compatibility shim - NOT a change to STAR-KV's methodology.

The `datasets` version resolved by an unpinned `pip install datasets` rejects
the bare legacy dataset id "wikitext" that STAR-KV/eval.py's _get_ppl_data()
uses (newer datasets requires a namespaced "org/name" id). Monkeypatches
datasets.load_dataset to redirect known legacy ids to their current namespaced
mirror before delegating to the real implementation, falling back to the
original id if the alias fails for any reason.

Import this BEFORE importing or runpy-executing eval.py or train.py, since
both do `from datasets import load_dataset` at module load time.

Also, if REPRO_GRAD_CKPT=1, patches LlamaForCausalLM.from_pretrained to enable
gradient checkpointing on every loaded model. train.py has no gradient-
checkpointing support (see Discrepancies Log) and OOMs on the single RTX PRO
6000 the ground-truth docs call for (authors used 2x for this step; the
single-GPU activation-memory estimate in SKILLS.md undercounts full backward
storage at seq_len 8192). Checkpointing is mathematically identical to a plain
backward pass (recompute vs. store), so this does not change training results
- it only trades compute for memory. No-op for the teacher (eval mode, no_grad).
"""
import os

import datasets

_LEGACY_DATASET_ALIASES = {
    "wikitext": "Salesforce/wikitext",
}

_original_load_dataset = datasets.load_dataset


def _patched_load_dataset(path, *args, **kwargs):
    alias = _LEGACY_DATASET_ALIASES.get(path)
    if alias is None:
        return _original_load_dataset(path, *args, **kwargs)
    print(f"[compat_shim] redirecting load_dataset('{path}') -> '{alias}'")
    try:
        return _original_load_dataset(alias, *args, **kwargs)
    except Exception as e:
        print(f"[compat_shim] alias '{alias}' failed ({e!r}); retrying original '{path}'")
        return _original_load_dataset(path, *args, **kwargs)


datasets.load_dataset = _patched_load_dataset

if os.environ.get("REPRO_GRAD_CKPT") == "1":
    import transformers

    _original_from_pretrained = transformers.LlamaForCausalLM.from_pretrained.__func__

    @classmethod
    def _patched_from_pretrained(cls, *args, **kwargs):
        model = _original_from_pretrained(cls, *args, **kwargs)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        print(f"[compat_shim] gradient checkpointing enabled on {cls.__name__} instance "
              "(REPRO_GRAD_CKPT=1; no-op for models in .eval() mode)")
        return model

    transformers.LlamaForCausalLM.from_pretrained = _patched_from_pretrained
