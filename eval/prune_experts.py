#!/usr/bin/env python3
"""Write a smaller EXL3 checkpoint with the least-used experts physically removed from every layer.

  prune_experts.py <src_exl3_dir> <dst_dir> <router_stats.npz> <drop_pct> [--drop-vision]

Every layer keeps the same number of experts (its most-used ones), renumbered 0..n-1; the router matrix keeps
only the matching rows; config.json's num_experts is lowered. Nothing is re-quantized: kept tensors are copied
byte for byte. The source directory is not modified.
"""
import json, os, re, struct, sys
from pathlib import Path
import numpy as np

args = [a for a in sys.argv[1:] if not a.startswith("--")]
drop_vision = "--drop-vision" in sys.argv
src, dst, stats, pct = Path(args[0]).resolve(), Path(args[1]).resolve(), np.load(args[2]), float(args[3])
counts = stats["counts"]; L, NE = counts.shape
keep_n = NE - int(round(NE * pct / 100))
# kept experts per layer, in original index order, and their new index
keep = [sorted(np.argsort(-counts[i], kind="stable")[:keep_n].tolist()) for i in range(L)]
newidx = [{old: new for new, old in enumerate(k)} for k in keep]
EXP = re.compile(r"^(model\.language_model\.layers\.(\d+)\.mlp\.experts\.)(\d+)(\..+)$")
GATE = re.compile(r"^model\.language_model\.layers\.(\d+)\.mlp\.gate\.weight$")


def remap(key):
    """Return the new tensor name, or None to drop it."""
    if drop_vision and ("visual" in key or "vision" in key):
        return None
    m = EXP.match(key)
    if not m:
        return key
    layer, old = int(m.group(2)), int(m.group(3))
    if layer >= L:
        return key
    if old not in newidx[layer]:
        return None
    return f"{m.group(1)}{newidx[layer][old]}{m.group(4)}"


dst.mkdir(parents=True, exist_ok=False)
index, kept_bytes, dropped_bytes = {}, 0, 0
for f in sorted(src.iterdir()):
    if f.name.startswith(".") or f.is_dir():
        continue
    if f.suffix != ".safetensors":
        if f.name not in ("model.safetensors.index.json", "config.json", "quantization_config.json"):
            os.symlink(f, dst / f.name)
        continue
    with open(f, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]; hdr = json.loads(fh.read(n)); base = 8 + n
        meta = hdr.pop("__metadata__", None)
        plan, off = [], 0
        for key, v in sorted(hdr.items(), key=lambda kv: kv[1]["data_offsets"][0]):
            new = remap(key); a, b = v["data_offsets"]
            if new is None:
                dropped_bytes += b - a; continue
            g = GATE.match(key)
            if g and int(g.group(1)) < L:
                rows = keep[int(g.group(1))]; width = v["shape"][1]; item = (b - a) // (v["shape"][0] * width)
                size = len(rows) * width * item
                plan.append((new, {**v, "shape": [len(rows), width], "data_offsets": [off, off + size]}, ("rows", a, rows, width * item)))
                dropped_bytes += (b - a) - size
            else:
                size = b - a
                plan.append((new, {**v, "data_offsets": [off, off + size]}, ("copy", a, b)))
            off += size; kept_bytes += size; index[new] = f.name
        out_hdr = {k: v for k, v, _ in plan}
        if meta is not None:
            out_hdr["__metadata__"] = meta
        raw = json.dumps(out_hdr, separators=(",", ":")).encode(); raw += b" " * ((8 - len(raw) % 8) % 8)
        with open(dst / f.name, "wb") as out:
            out.write(struct.pack("<Q", len(raw))); out.write(raw)
            for _, _, op in plan:
                if op[0] == "copy":
                    fh.seek(base + op[1]); left = op[2] - op[1]
                    while left:
                        chunk = fh.read(min(left, 64 << 20)); out.write(chunk); left -= len(chunk)
                else:
                    _, a, rows, rowbytes = op
                    for r in rows:
                        fh.seek(base + a + r * rowbytes); out.write(fh.read(rowbytes))

cfg = json.loads((src / "config.json").read_text()); cfg["text_config"]["num_experts"] = keep_n
(dst / "config.json").write_text(json.dumps(cfg, indent=4))
q = json.loads((src / "quantization_config.json").read_text()); ts = {}
for mod, info in q["tensor_storage"].items():
    probe = remap(mod + ".x")
    if probe is None:
        continue
    newmod = probe[:-2]
    st = {}
    for t, d in info.get("stored_tensors", {}).items():
        nt = remap(t)
        if nt is None:
            continue
        g = GATE.match(t)
        if g and int(g.group(1)) < L:
            d = {**d, "shape": [keep_n, d["shape"][1]], "n_bytes": d["n_bytes"] * keep_n // d["shape"][0]}
        st[nt] = d
    ts[newmod] = {**info, "stored_tensors": st}
q["tensor_storage"] = ts
(dst / "quantization_config.json").write_text(json.dumps(q))
ix = src / "model.safetensors.index.json"
if ix.exists():
    d = json.loads(ix.read_text()); d["weight_map"] = index; (dst / ix.name).write_text(json.dumps(d, indent=2))
(dst / "pruning.json").write_text(json.dumps({"source": src.name, "drop_pct": pct, "experts_kept_per_layer": keep_n, "of": NE,
                                              "vision_tower_removed": drop_vision, "kept_original_indices": keep}))
print(json.dumps({"dst": str(dst), "experts_per_layer": keep_n, "weights_gib": round(kept_bytes / 2**30, 3), "removed_gib": round(dropped_bytes / 2**30, 3)}))
