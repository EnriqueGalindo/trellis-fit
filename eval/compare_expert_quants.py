#!/usr/bin/env python3
"""Output error of quantized expert matrices on held-out activations, for several quants of the same modules.

  compare_expert_quants.py <bf16_dir> <heldout.npz> <label=exl3_dir> [<label=exl3_dir> ...]

For every module key in the npz: rel. output error = ||X (Wq - W)|| / ||X W||, and plain weight error ||Wq - W|| / ||W||.
"""
import json, re, sys
from collections import defaultdict
import numpy as np
from ponyexl3.ref.loader import load_exl3_layer
from ponyexl3.ref.reconstruct import reconstruct_public_weights
from ponyexl3.convert.fixtures import resolve_source_linear, read_source_public_block

bf16, acts = sys.argv[1], np.load(sys.argv[2])
quants = dict(a.split("=", 1) for a in sys.argv[3:])
res = defaultdict(lambda: defaultdict(list))
for key in acts.files:
    X = acts[key].astype(np.float32)
    src = resolve_source_linear(bf16, key)
    W = np.asarray(read_source_public_block(bf16, src, in_start=0, out_start=0, rows=src.in_features, cols=src.out_features), dtype=np.float32)  # (in, out)
    ref_out = X @ W
    proj = key.rsplit(".", 1)[1]
    for label, d in quants.items():
        L = load_exl3_layer(d, key)
        Wq = np.asarray(reconstruct_public_weights(L.trellis, L.suh, L.svh, L.k, mcg=bool(L.mcg), mul1=bool(L.mul1)), dtype=np.float32)
        if Wq.shape != W.shape:
            Wq = Wq.T
        res[label][proj].append((float(np.linalg.norm(X @ Wq - ref_out) / np.linalg.norm(ref_out)), float(np.linalg.norm(Wq - W) / np.linalg.norm(W))))
out = {}
for label, by in res.items():
    out[label] = {p: {"modules": len(v), "output_rel_err": round(float(np.mean([a for a, _ in v])), 4), "weight_rel_err": round(float(np.mean([b for _, b in v])), 4)} for p, v in by.items()}
print(json.dumps(out, indent=1))
