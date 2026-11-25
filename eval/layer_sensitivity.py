#!/usr/bin/env python3
"""Which layers' experts tolerate 1 bit? (Track A: per-layer bit allocation, loadable by today's runtimes.)

  layer_sensitivity.py <bf16_model_dir> <reference_name>

1. Sweep: for each layer in turn, put ALL routed experts of that layer at 1 bit (stand-in quantizer: sign x row
   scale), score KLD vs the bf16 reference, then restore the layer.
2. Cumulative: apply 1 bit to the 5, 10, 15, 20 least sensitive layers together and score.
3. Same sets with every other layer's experts at the 2-bit stand-in, to see whether costs add.

The stand-in has about 1.3x the error of a real 1-bit trellis and no calibration, so costs here are upper-ish bounds.
"""
import json, sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm.utils import load_model, load_tokenizer

import evallib as E

model_dir, ref_name = Path(sys.argv[1]).resolve(), sys.argv[2]
model, _ = load_model(model_dir)
tok = load_tokenizer(model_dir)
all_layers = model.language_model.model.layers
moe = [(i, l) for i, l in enumerate(all_layers) if hasattr(l.mlp, "switch_mlp")]
ref_dir = E.REF_DIR / ref_name
ids, bos = E.wiki_tokens(tok)
wins = E.windows(ids, bos)
assert json.loads((ref_dir / "meta.json").read_text())["tokens_sha"] == E.tokens_sha(wins)
first, last = E.scored_range()
refs = None
NAMES = ("gate_proj", "up_proj", "down_proj")


def q1(w):
    a = mx.mean(mx.abs(w), axis=-1, keepdims=True)
    return mx.where(w >= 0, a, -a)


def q2(w):
    s = mx.sqrt(mx.mean(w * w, axis=-1, keepdims=True))
    mag = mx.where(mx.abs(w) > 0.9816 * s, 1.510 * s, 0.4528 * s)
    return mx.where(w >= 0, mag, -mag)


def apply(layer, fn):
    saved = {}
    for n in NAMES:
        p = getattr(layer.mlp.switch_mlp, n)
        saved[n] = p.weight
        p.weight = fn(p.weight.astype(mx.float32)).astype(mx.bfloat16)
        mx.eval(p.weight)
    return saved


def restore(layer, saved):
    for n in NAMES:
        getattr(layer.mlp.switch_mlp, n).weight = saved[n]
    mx.clear_cache()


def score():
    nll, n, kl_sum, agree = 0.0, 0, 0.0, 0
    for i, w in enumerate(wins):
        lp = nn.log_softmax(model(mx.array([w]))[0, first:last, :].astype(mx.float32), axis=-1)
        tgt = mx.array(w[first + 1 : last + 1])
        ref = mx.array(np.load(ref_dir / f"{i:02d}.npy")).astype(mx.float32)
        kl = mx.sum(mx.exp(ref) * (ref - lp))
        a = mx.sum(mx.argmax(ref, axis=-1) == mx.argmax(lp, axis=-1))
        s = -mx.sum(mx.take_along_axis(lp, tgt[:, None], axis=-1))
        mx.eval(kl, a, s)
        nll += float(s.item()); n += int(tgt.shape[0]); kl_sum += float(kl.item()); agree += int(a.item())
        mx.clear_cache()
    return {"ppl": round(float(np.exp(nll / n)), 3), "kld_mean": round(kl_sum / n, 4), "top1_agree_pct": round(100 * agree / n, 1)}


out = {"model": model_dir.name, "sweep": [], "cumulative": []}
for idx, layer in moe:
    saved = apply(layer, q1)
    r = {"layer": idx, **score()}
    restore(layer, saved)
    print(json.dumps(r), flush=True); out["sweep"].append(r)

order = [r["layer"] for r in sorted(out["sweep"], key=lambda r: r["kld_mean"])]
out["order_least_sensitive_first"] = order
by_idx = dict(moe)
done = 0
for n in (5, 10, 15, 20):
    for idx in order[done:n]:
        apply(by_idx[idx], q1)
    done = n
    r = {"layers_at_1_bit": n, "rest": "untouched", "which": sorted(order[:n]), **score()}
    print(json.dumps(r), flush=True); out["cumulative"].append(r)
    (E.RAW_DIR / f"layer_sensitivity__{model_dir.name}.json").write_text(json.dumps(out, indent=1))

# reload a clean model for the 2-bit-background check
del model
mx.clear_cache()
model, _ = load_model(model_dir)
all_layers = model.language_model.model.layers
by_idx = {i: l for i, l in enumerate(all_layers) if hasattr(l.mlp, "switch_mlp")}
for n in (0, 10, 20):
    for idx, layer in by_idx.items():
        pass
    if n == 0:
        for idx, layer in by_idx.items():
            apply(layer, q2)
    else:
        # layers in order[prev:n] go from 2-bit back to original then to 1-bit: reload is simpler than undoing
        del model; mx.clear_cache()
        model, _ = load_model(model_dir)
        all_layers = model.language_model.model.layers
        by_idx = {i: l for i, l in enumerate(all_layers) if hasattr(l.mlp, "switch_mlp")}
        for idx, layer in by_idx.items():
            apply(layer, q1 if idx in order[:n] else q2)
    r = {"layers_at_1_bit": n, "rest": "2-bit stand-in", **score()}
    print(json.dumps(r), flush=True); out["cumulative"].append(r)
(E.RAW_DIR / f"layer_sensitivity__{model_dir.name}.json").write_text(json.dumps(out, indent=1))
