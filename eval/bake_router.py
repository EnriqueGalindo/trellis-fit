#!/usr/bin/env python3
"""Make a copy of a quantized model in which the least-used experts can never be selected.

  bake_router.py <src (EXL3 dir | .gguf)> <dst> <router_stats.npz> <drop_pct>

Only the small router matrices change: each dropped expert's row becomes -LAMBDA x (mean router-input
direction), which makes its score hugely negative for every token. Expert weights are untouched, so the
copy is the same size; a runtime that maps weights from disk simply never reads the dropped experts.
EXL3 dirs run with any Python that has numpy; GGUF needs the `gguf` package (uv run --with gguf).
"""
import json, os, re, shutil, struct, sys
from pathlib import Path
import numpy as np

src, dst, stats, pct = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve(), np.load(sys.argv[3]), int(sys.argv[4])
LAMBDA = 50.0
counts, mean_dir = stats["counts"], stats["mean_dir"]
L, NE = counts.shape
k = int(round(NE * pct / 100))
drop = np.zeros((L, NE), dtype=bool)
for i in range(L):
    drop[i, np.argsort(counts[i])[:k]] = True
unit = mean_dir / np.linalg.norm(mean_dir, axis=1, keepdims=True)


def to_bf16(a):
    u = a.astype(np.float32).view(np.uint32)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def patch(path, offset, shape, dtype, layer):
    """Overwrite dropped rows of one (NE, hidden) router matrix stored at `offset`."""
    assert tuple(shape) == (NE, unit.shape[1]), (shape, layer)
    rows = (-LAMBDA * unit[layer]).astype(np.float32)
    np_dtype = {"F32": np.float32, "F16": np.float16, "BF16": np.uint16}[dtype]
    m = np.memmap(path, dtype=np_dtype, mode="r+", offset=offset, shape=tuple(shape))
    m[drop[layer]] = to_bf16(rows) if dtype == "BF16" else rows.astype(np_dtype)
    m.flush(); del m


patched = 0
if src.is_dir():
    dst.mkdir(parents=True, exist_ok=False)
    for f in sorted(src.iterdir()):
        if f.name.startswith(".") or f.is_dir():
            continue
        if f.suffix != ".safetensors":
            os.symlink(f, dst / f.name); continue
        shutil.copyfile(f, dst / f.name)
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]; hdr = json.loads(fh.read(n))
        for key, v in hdr.items():
            m = re.fullmatch(r"model\.language_model\.layers\.(\d+)\.mlp\.gate\.weight", key)
            if m:
                patch(dst / f.name, 8 + n + v["data_offsets"][0], v["shape"], v["dtype"], int(m.group(1))); patched += 1
else:
    from gguf import GGUFReader
    shutil.copyfile(src, dst)
    for t in GGUFReader(str(src)).tensors:
        m = re.fullmatch(r"blk\.(\d+)\.ffn_gate_inp\.weight", t.name)
        if m and int(m.group(1)) < L:  # layers beyond the main stack (the MTP head) keep their router
            patch(dst, int(t.data_offset), tuple(int(x) for x in t.shape)[::-1], t.tensor_type.name, int(m.group(1))); patched += 1
print(json.dumps({"dst": str(dst), "router_matrices_patched": patched, "experts_dropped_per_layer": k, "of": NE}))
assert patched == L, f"expected {L} router matrices"
