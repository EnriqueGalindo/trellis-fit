#!/usr/bin/env python3
"""Where the bytes are in an EXL3 checkpoint. Read-only: parses safetensors headers.

  byte_breakdown.py <exl3_dir> [<bf16_dir>]

With the bf16 source given, also reports parameters per group and effective bits per weight.
"""
import json, re, struct, sys
from collections import defaultdict
from pathlib import Path

GIB = 1024**3
EXL3_SFX = ("trellis", "suh", "svh", "su", "sv", "mcg", "mul1", "bias")


def headers(d):
    out = {}
    for f in sorted(Path(d).glob("*.safetensors")):
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            h = json.loads(fh.read(n))
        for k, v in h.items():
            if k != "__metadata__":
                out[k] = v
    return out


def nbytes(v):
    return v["data_offsets"][1] - v["data_offsets"][0]


def nparams(v):
    n = 1
    for s in v["shape"]:
        n *= s
    return n


def module_of(key):
    """Strip an EXL3 sub-tensor suffix, and .weight, to get the module name."""
    base, _, last = key.rpartition(".")
    if last in EXL3_SFX or last == "weight":
        return base, last
    return key, ""


def group(mod):
    m = mod
    if "visual" in m or "vision" in m: return "vision tower"
    if "embed_tokens" in m: return "token embeddings"
    if "lm_head" in m: return "output head"
    if "mtp" in m: return "MTP head"
    if ".mlp.experts." in m or "switch_mlp" in m: return "routed experts (MoE)"
    if "shared_expert" in m and "gate" not in m.split("shared_expert")[-1][:6]: return "shared expert"
    if "shared_expert" in m: return "shared expert"
    if re.search(r"\.mlp\.gate$", m): return "MoE router"
    if "linear_attn" in m: return "linear attention"
    if "self_attn" in m: return "full attention"
    if "norm" in m: return "norms"
    return "other"


def main():
    q = headers(sys.argv[1])
    gb, sub = defaultdict(int), defaultdict(int)
    mods = defaultdict(set)
    for k, v in q.items():
        mod, sfx = module_of(k)
        g = group(mod)
        gb[g] += nbytes(v)
        mods[g].add(mod)
        sub[(g, sfx if sfx in EXL3_SFX else f"plain {v['dtype']}")] += nbytes(v)
    total = sum(gb.values())

    gp = defaultdict(int)
    if len(sys.argv) > 2:
        for k, v in headers(sys.argv[2]).items():
            gp[group(module_of(k)[0])] += nparams(v)

    print(f"total weight bytes: {total:,} ({total / GIB:.3f} GiB)\n")
    print("| Group | Modules | GiB | Share | Params (B) | Effective bpw |")
    print("|---|---|---|---|---|---|")
    for g, b in sorted(gb.items(), key=lambda x: -x[1]):
        p = gp.get(g)
        print(f"| {g} | {len(mods[g])} | {b / GIB:.3f} | {100 * b / total:.1f}% | "
              f"{(p / 1e9):.3f} | {(b * 8 / p):.2f} |" if p else
              f"| {g} | {len(mods[g])} | {b / GIB:.3f} | {100 * b / total:.1f}% | n/a | n/a |")
    missing = {g: p for g, p in gp.items() if g not in gb}
    if missing:
        print("\nin the bf16 source but absent from the quant:", {g: f"{p / 1e9:.3f} B params" for g, p in missing.items()})

    print("\nBy tensor kind inside each group (GiB):")
    for (g, s), b in sorted(sub.items(), key=lambda x: (-gb[x[0][0]], -x[1])):
        if b / GIB >= 0.001:
            print(f"  {g:24s} {s:16s} {b / GIB:8.3f}")

    qc = Path(sys.argv[1]) / "quantization_config.json"
    if qc.exists():
        ts = json.loads(qc.read_text()).get("tensor_storage", {})
        kb = defaultdict(lambda: defaultdict(int))
        for name, info in ts.items():
            k = info.get("bits_per_weight", info.get("quant_format", {}).get("bits_per_weight") if isinstance(info.get("quant_format"), dict) else None)
            if k is None:
                k = info.get("bits", info.get("K"))
            kb[group(name)][k] += 1
        print("\nBit width per module (count of modules at each K):")
        for g, d in kb.items():
            print(f"  {g:24s} {dict(sorted(d.items(), key=lambda x: str(x[0])))}")
        ex = next(iter(ts.items()))
        print("\nsample tensor_storage entry keys:", [k for k in ex[1].keys()])


if __name__ == "__main__":
    main()
