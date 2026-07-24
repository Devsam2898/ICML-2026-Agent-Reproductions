"""Environment-compatibility shim - NOT a change to STAR-KV's methodology.

The `datasets` version resolved by an unpinned `pip install datasets` rejects
the bare legacy dataset id "wikitext" that STAR-KV/eval.py's _get_ppl_data()
uses (newer datasets requires a namespaced "org/name" id). Monkeypatches
datasets.load_dataset to redirect known legacy ids to their current namespaced
mirror before delegating to the real implementation, falling back to the
original id if the alias fails for any reason.

Import this BEFORE importing or runpy-executing eval.py or train.py, since
both do `from datasets import load_dataset` at module load time.
"""
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
