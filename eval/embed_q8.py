#!/usr/bin/env python3
"""Write a copy of an EXL3 checkpoint whose token-embedding table is stored at 8 bits per value.

  embed_q8.py <src_exl3_dir> <dst_dir>

`model.language_model.embed_tokens.weight` (bf16/f16, [vocab, hidden]) is replaced by
  model.language_model.embed_tokens.q8        int8  [vocab, hidden]   round(w / scale), clamped to +-127
  model.language_model.embed_tokens.q8_scale  f16   [vocab, 1]        row max-abs / 127
Everything else is copied byte for byte. Needs the MLXL3 fork (branch fit-levers); stock engines will not load it.
The source directory is not modified.
"""
import json, os, struct, sys
from pathlib import Path
import numpy as np

src, dst = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
NAME = "model.language_model.embed_tokens.weight"
Q8, QS = NAME[:-6] + "q8", NAME[:-6] + "q8_scale"
dst.mkdir(parents=True, exist_ok=False)
index, done, stats = {}, False, {}
for f in sorted(src.iterdir()):
    if f.name.startswith(".") or f.is_dir():
        continue
    if f.suffix != ".safetensors":
        if f.name not in ("model.safetensors.index.json", "quantization_config.json"):
            os.symlink(f.resolve(), dst / f.name)
        continue
    with open(f, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]; hdr = json.loads(fh.read(n)); base = 8 + n
        meta = hdr.pop("__metadata__", None)
        if NAME not in hdr:
            os.link(f.resolve(), dst / f.name) if os.stat(f.resolve()).st_dev == os.stat(dst).st_dev else os.symlink(f.resolve(), dst / f.name)
            index.update({k: f.name for k in hdr}); continue
        plan, off = [], 0
        for key, v in sorted(hdr.items(), key=lambda kv: kv[1]["data_offsets"][0]):
            a, b = v["data_offsets"]
            if key != NAME:
                plan.append((key, {**v, "data_offsets": [off, off + b - a]}, ("copy", a, b))); off += b - a; index[key] = f.name
                continue
            rows, width = v["shape"]
            fh.seek(base + a); raw = np.frombuffer(fh.read(b - a), dtype=np.uint16)
            if v["dtype"] == "BF16":
                w = (raw.astype(np.uint32) << 16).view(np.float32)
            elif v["dtype"] == "F16":
                w = raw.view(np.float16).astype(np.float32)
            else:
                raise SystemExit(f"unexpected embedding dtype {v['dtype']}")
            w = w.reshape(rows, width)
            peak = np.abs(w).max(axis=1, keepdims=True)
            scale = np.where(peak > 0, peak / 127.0, 1.0).astype(np.float16)   # the stored f16 scale is what decoding uses
            codes = np.clip(np.rint(w / scale.astype(np.float32)), -127, 127).astype(np.int8)
            err = codes.astype(np.float32) * scale.astype(np.float32) - w
            stats = {"rows": rows, "width": width, "rel_rmse": float(np.sqrt((err ** 2).mean() / (w ** 2).mean())), "bytes_before": b - a,
                     "bytes_after": codes.nbytes + scale.nbytes}
            for nm, arr, dt in ((Q8, codes, "I8"), (QS, scale, "F16")):
                plan.append((nm, {"dtype": dt, "shape": list(arr.shape), "data_offsets": [off, off + arr.nbytes]}, ("bytes", arr.tobytes())))
                off += arr.nbytes; index[nm] = f.name
            done = True
        out_hdr = {k: v for k, v, _ in plan}
        if meta is not None:
            out_hdr["__metadata__"] = meta
        rawh = json.dumps(out_hdr, separators=(",", ":")).encode(); rawh += b" " * ((8 - len(rawh) % 8) % 8)
        with open(dst / f.name, "wb") as out:
            out.write(struct.pack("<Q", len(rawh))); out.write(rawh)
            for _, _, op in plan:
                if op[0] == "bytes":
                    out.write(op[1]); continue
                fh.seek(base + op[1]); left = op[2] - op[1]
                while left:
                    chunk = fh.read(min(left, 64 << 20)); out.write(chunk); left -= len(chunk)
assert done, f"{NAME} not found"
q = json.loads((src / "quantization_config.json").read_text())
mod = NAME[:-7]
q["tensor_storage"][mod] = {"stored_tensors": {
    Q8: {"shape": [stats["rows"], stats["width"]], "n_bytes": stats["rows"] * stats["width"], "dtype": "torch.int8"},
    QS: {"shape": [stats["rows"], 1], "n_bytes": stats["rows"] * 2, "dtype": "torch.float16"}}, "embedding_format": "q8_row_scale"}
(dst / "quantization_config.json").write_text(json.dumps(q))
ix = src / "model.safetensors.index.json"
if ix.exists():
    d = json.loads(ix.read_text()); d["weight_map"] = index; (dst / ix.name).write_text(json.dumps(d, indent=2))
print(json.dumps({"dst": str(dst), **stats, "saved_gib": round((stats["bytes_before"] - stats["bytes_after"]) / 2**30, 3)}))
