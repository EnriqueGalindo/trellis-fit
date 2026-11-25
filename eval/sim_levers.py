#!/usr/bin/env python3
"""Simulate size-saving levers on the bf16 model in mlx-lm and score each against the bf16 reference.

  sim_levers.py <bf16_model_dir> <reference_name> <config> [<config> ...]

Configs (each runs on a freshly loaded model, in this process, one after another is NOT supported: one config
per invocation keeps memory flat; pass several and the script re-executes itself per config):
  base            nothing changed (sanity: KLD must be ~0)
  emb8|emb6|emb4  token embeddings rounded to N bits per value (per-row uniform grid)
  x2              every routed expert weight to a 2-bit scalar quantizer (proxy for a 2 bpw quant, no calibration)
  x1cNN           the least-used NN% of experts to a 1-bit quantizer (sign x row scale), the rest untouched
  x2+x1cNN        least-used NN% at 1 bit, the rest at 2 bits
  bakeNN          drop the least-used NN% by editing router WEIGHTS only (row := -LAMBDA * mean input direction),
                  and report how often a dropped expert is still selected
  +combine with '+', e.g. emb8+x2+x1c25

The scalar quantizers are stand-ins: real EXL3 uses trellis coding with calibration and does better at equal bits.
Use the x2 row to see how pessimistic the stand-in is (the real 2.08 bpw quant scores KLD 0.114).
"""
import json, os, subprocess, sys
from pathlib import Path

import numpy as np

import evallib as E

model_dir, ref_name, configs = Path(sys.argv[1]).resolve(), sys.argv[2], sys.argv[3:]
if len(configs) > 1:
    for c in configs:
        subprocess.run([sys.executable, __file__, str(model_dir), ref_name, c], check=False)
    sys.exit(0)
cfg = configs[0]
parts = cfg.split("+")

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models import switch_layers
from mlx_lm.utils import load_model, load_tokenizer

model, _ = load_model(model_dir)
tok = load_tokenizer(model_dir)
hf = getattr(tok, "_tokenizer", tok)
tm = model.language_model.model
layers = [l for l in tm.layers if hasattr(l.mlp, "switch_mlp")]
L, NE = len(layers), layers[0].mlp.num_experts
ref_dir = E.REF_DIR / ref_name
CAL = [E.EVAL_DIR / "wikitext-dl/wikitext-2-raw/wiki.train.raw", ref_dir / "code_calib.txt"]


def calib_ids():
    for p in CAL:
        if p.exists():
            ids = list(hf.encode(p.read_text(encoding="utf-8", errors="replace"), add_special_tokens=False))[:16384]
            for s in range(0, len(ids), 2048):
                yield ids[s : s + 2048]


# ---- usage counts + mean router-input direction (cached next to the reference)
cache = ref_dir / "router_stats.npz"
if cache.exists():
    z = np.load(cache); counts, mean_dir = z["counts"], z["mean_dir"]
else:
    counts = np.zeros((L, NE), dtype=np.int64)
    sums = [None] * L; ntok = 0
    ids_of = {id(l.mlp.switch_mlp): i for i, l in enumerate(layers)}
    orig = switch_layers.SwitchGLU.__call__

    def spy(self, x, indices):
        li = ids_of.get(id(self))
        if li is not None:
            counts[li] += np.bincount(np.array(indices).ravel(), minlength=NE)
            xn = x.astype(mx.float32).reshape(-1, x.shape[-1])
            xn = xn / mx.linalg.norm(xn, axis=-1, keepdims=True)
            s = np.array(mx.sum(xn, axis=0)); sums[li] = s if sums[li] is None else sums[li] + s
        return orig(self, x, indices)

    switch_layers.SwitchGLU.__call__ = spy
    for w in calib_ids():
        mx.eval(model(mx.array([w]))); ntok += len(w)
    switch_layers.SwitchGLU.__call__ = orig
    mean_dir = np.stack(sums) / ntok
    np.savez(cache, counts=counts, mean_dir=mean_dir)

order = np.argsort(counts, axis=1)  # ascending: coldest first, per layer


def cold_mask(pct):
    k = int(round(NE * pct / 100)); m = np.zeros((L, NE), dtype=bool)
    for i in range(L):
        m[i, order[i, :k]] = True
    return m


def q1(w):  # sign x mean|w| per output row
    a = mx.mean(mx.abs(w), axis=-1, keepdims=True)
    return mx.where(w >= 0, a, -a)


def q2(w):  # 4-level Lloyd-Max for a Gaussian, per output row
    s = mx.sqrt(mx.mean(w * w, axis=-1, keepdims=True))
    mag = mx.where(mx.abs(w) > 0.9816 * s, 1.510 * s, 0.4528 * s)
    return mx.where(w >= 0, mag, -mag)


def requant_experts(fn, mask):  # mask: bool (L, NE) of experts to change
    for i, l in enumerate(layers):
        if not mask[i].any():
            continue
        sel = mx.array(mask[i])[:, None, None]
        for name in ("gate_proj", "up_proj", "down_proj"):
            p = getattr(l.mlp.switch_mlp, name)
            w = p.weight.astype(mx.float32)
            p.weight = mx.where(sel, fn(w), w).astype(mx.bfloat16)
            mx.eval(p.weight)
        mx.clear_cache()


note = {}
for part in parts:
    if part == "base":
        pass
    elif part.startswith("emb"):
        bits = int(part[3:]); w = tm.embed_tokens.weight.astype(mx.float32)
        lo, hi = mx.min(w, axis=-1, keepdims=True), mx.max(w, axis=-1, keepdims=True)
        step = (hi - lo) / (2**bits - 1)
        tm.embed_tokens.weight = (mx.round((w - lo) / step) * step + lo).astype(mx.bfloat16); mx.eval(tm.embed_tokens.weight)
    elif part == "x2":
        hot = ~cold_mask(int(next((p[3:] for p in parts if p.startswith("x1c")), 0)))
        requant_experts(q2, hot)
    elif part.startswith("x1c"):
        requant_experts(q1, cold_mask(int(part[3:])))
    elif part.startswith("bake"):
        m = cold_mask(int(part[4:])); LAMBDA = 1000.0
        for i, l in enumerate(layers):
            w = np.array(l.mlp.gate.weight.astype(mx.float32))
            u = mean_dir[i] / np.linalg.norm(mean_dir[i])
            w[m[i]] = -LAMBDA * u
            l.mlp.gate.weight = mx.array(w).astype(mx.bfloat16)
        note["mean_input_direction_strength_min"] = float(np.linalg.norm(mean_dir, axis=1).min())
        dropped = m; hits = [0, 0]
        ids_of = {id(l.mlp.switch_mlp): i for i, l in enumerate(layers)}
        orig = switch_layers.SwitchGLU.__call__

        def spy2(self, x, indices):
            li = ids_of.get(id(self))
            if li is not None:
                idx = np.array(indices).ravel(); hits[0] += int(dropped[li][idx].sum()); hits[1] += idx.size
            return orig(self, x, indices)

        switch_layers.SwitchGLU.__call__ = spy2
        note["_hits"] = hits
    else:
        raise SystemExit(f"unknown config part {part}")

ids, bos = E.wiki_tokens(tok)
wins = E.windows(ids, bos)
assert json.loads((ref_dir / "meta.json").read_text())["tokens_sha"] == E.tokens_sha(wins)
first, last = E.scored_range()
nll, n, klds, agree = 0.0, 0, [], 0
for i, w in enumerate(wins):
    lp = nn.log_softmax(model(mx.array([w]))[0, first:last, :].astype(mx.float32), axis=-1)
    tgt = mx.array(w[first + 1 : last + 1])
    ref = mx.array(np.load(ref_dir / f"{i:02d}.npy")).astype(mx.float32)
    kl = mx.sum(mx.exp(ref) * (ref - lp), axis=-1)
    a = mx.sum(mx.argmax(ref, axis=-1) == mx.argmax(lp, axis=-1))
    s = -mx.sum(mx.take_along_axis(lp, tgt[:, None], axis=-1))
    mx.eval(kl, a, s)
    nll += float(s.item()); n += int(tgt.shape[0]); klds.extend(kl.tolist()); agree += int(a.item())
    mx.clear_cache()
row = {"config": cfg, "ppl": round(float(np.exp(nll / n)), 3), "kld_mean": round(float(np.mean(klds)), 4),
       "kld_p99": round(float(np.quantile(klds, 0.99)), 3), "top1_agree_pct": round(100 * agree / n, 1)}
if "_hits" in note:
    h = note.pop("_hits"); row["dropped_expert_selected_pct"] = round(100 * h[0] / max(h[1], 1), 4)
row.update(note)
print(json.dumps(row), flush=True)
with open(E.RAW_DIR / f"sim_levers__{model_dir.name}.jsonl", "a") as f:
    f.write(json.dumps(row) + "\n")
