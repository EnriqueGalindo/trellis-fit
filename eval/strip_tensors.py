#!/usr/bin/env python3
"""Copy a checkpoint directory, leaving out tensors whose name matches a regex.

  strip_tensors.py <src_dir> <dst_dir> <regex>

Weights are rewritten shard by shard (streamed, no model load); every other file is
symlinked. The source directory is never modified.
"""
import json, os, re, struct, sys
from pathlib import Path

src, dst, pat = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve(), re.compile(sys.argv[3])
dst.mkdir(parents=True, exist_ok=False)
dropped, dropped_bytes, index = [], 0, {}
for f in sorted(src.iterdir()):
    if f.name.startswith(".") or f.is_dir():
        continue
    if f.suffix != ".safetensors":
        if f.name != "model.safetensors.index.json":
            os.symlink(f, dst / f.name)
        continue
    with open(f, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        hdr = json.loads(fh.read(n))
        base = 8 + n
        meta = hdr.pop("__metadata__", None)
        keep = sorted(((k, v) for k, v in hdr.items() if not pat.search(k)), key=lambda kv: kv[1]["data_offsets"][0])
        for k, v in hdr.items():
            if pat.search(k):
                dropped.append(k); dropped_bytes += v["data_offsets"][1] - v["data_offsets"][0]
        new, off = {}, 0
        for k, v in keep:
            size = v["data_offsets"][1] - v["data_offsets"][0]
            new[k] = {**v, "data_offsets": [off, off + size]}; off += size
            index[k] = f.name
        if meta is not None:
            new["__metadata__"] = meta
        raw = json.dumps(new, separators=(",", ":")).encode()
        raw += b" " * ((8 - len(raw) % 8) % 8)
        with open(dst / f.name, "wb") as out:
            out.write(struct.pack("<Q", len(raw))); out.write(raw)
            for k, v in keep:
                a, b = v["data_offsets"]
                fh.seek(base + a); left = b - a
                while left:
                    chunk = fh.read(min(left, 64 << 20)); out.write(chunk); left -= len(chunk)
ix = src / "model.safetensors.index.json"
if ix.exists():
    d = json.loads(ix.read_text()); d["weight_map"] = {k: v for k, v in d.get("weight_map", {}).items() if not pat.search(k)}
    (dst / ix.name).write_text(json.dumps(d, indent=2))
print(json.dumps({"dropped_tensors": len(dropped), "dropped_gib": dropped_bytes / 1024**3, "example": dropped[:2]}))
