"""Deterministic entrypoint: seeds RNGs, then runs a STAR-KV script's __main__.

Usage: python seeded_run.py /path/to/train.py --model ... --output ...

Exists because STAR-KV/train.py has no seeding of its own (see Discrepancies
Log in the Trackio logbook). Does not modify any vendored STAR-KV/ file -
it only seeds the process before handing control to the target script.
"""
import os
import random
import runpy
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compat_shim  # noqa: E402,F401  (must import before eval.py/train.py)

SEED = int(os.environ.get("REPRO_SEED", "42"))
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
print(f"[seeded_run] SEED={SEED}")

target = sys.argv[1]
sys.argv = sys.argv[1:]
sys.path.insert(0, os.path.dirname(os.path.abspath(target)))
runpy.run_path(target, run_name="__main__")
