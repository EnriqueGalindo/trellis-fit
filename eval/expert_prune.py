#!/usr/bin/env python3
"""What happens to quality if the router may only use the most-used experts?

  expert_prune.py <bf16_model_dir> <reference_name> <calibration_text> [more calibration texts ...] [--held-out FILE]

1. Count how often each expert is picked on the calibration texts (never the WikiText-2 test set).
2. For several keep-fractions, mask the dropped experts out of every layer's router, so the top-8 are
   chosen among the kept ones, and score PPL + KLD against the unmodified bf16 reference on the standard
   WikiText-2 test windows. Optionally also PPL on a held-out text of another kind.

"per-layer" keeps the same share in every layer; "global" ranks all (layer, expert) pairs together.
Runs on the bf16 model, so this isolates the effect of dropping experts from the effect of quantization.
"""
import json, sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm.models import switch_layers
from mlx_lm.utils import load_model, load_tokenizer

import evallib as E

args = sys.argv[1:]
held = None
if "--held-out" in args:
    i = args.index("--held-out"); held = Path(args[i + 1]); del args[i : i + 2]
model_dir, ref_name, calib = Path(args[0]).resolve(), args[1], [Path(p) for p in args[2:]]
CAL_TOKENS, WINDOW = 16384, 2048

model, _ = load_model(model_dir)
tok = load_tokenizer(model_dir)
hf = getattr(tok, "_tokenizer", tok)
layers = [l for l in model.language_model.model.layers if hasattr(l.mlp, "switch_mlp")]
L, NE = len(layers), layers[0].mlp.num_experts


class MaskedGate(nn.Module):
    def __init__(self, inner):
        super().__init__()
        self.inner = inner
        self.mask = None

    def __call__(self, x):
        g = self.inner(x)
        return g if self.mask is None else g + self.mask.astype(g.dtype)


for l in layers:
    l.mlp.gate = MaskedGate(l.mlp.gate)

# ---- 1. usage on calibration text
ids_of = {id(l.mlp.switch_mlp): i for i, l in enumerate(layers)}
counts = np.zeros((L, NE), dtype=np.int64)
orig = switch_layers.SwitchGLU.__call__
recording = True


def spy(self, x, indices):
    li = ids_of.get(id(self))
    if recording and li is not None:
        counts[li] += np.bincount(np.array(indices).ravel(), minlength=NE)
    return orig(self, x, indices)


switch_layers.SwitchGLU.__call__ = spy
cal_tokens = 0
for path in calib:
    ids = list(hf.encode(path.read_text(encoding="utf-8", errors="replace"), add_special_tokens=False))[:CAL_TOKENS]
    for s in range(0, len(ids), WINDOW):
        mx.eval(model(mx.array([ids[s : s + WINDOW]])))
    cal_tokens += len(ids)
recording = False

# ---- 2. scoring
ids, bos = E.wiki_tokens(tok)
wins = E.windows(ids, bos)
ref_dir = E.REF_DIR / ref_name
assert json.loads((ref_dir / "meta.json").read_text())["tokens_sha"] == E.tokens_sha(wins), "reference is on other tokens"
first, last = E.scored_range()
held_wins = []
if held:
    h = list(hf.encode(held.read_text(encoding="utf-8", errors="replace"), add_special_tokens=False))
    held_wins = [h[i * WINDOW : (i + 1) * WINDOW] for i in range(min(4, len(h) // WINDOW))]


def score():
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
    out = {"ppl": float(np.exp(nll / n)), "kld_mean": float(np.mean(klds)), "kld_p99": float(np.quantile(klds, 0.99)), "top1_agree": agree / n}
    if held_wins:
        hn, hc = 0.0, 0
        for w in held_wins:
            lp = nn.log_softmax(model(mx.array([w]))[0, first:last, :].astype(mx.float32), axis=-1)
            s = -mx.sum(mx.take_along_axis(lp, mx.array(w[first + 1 : last + 1])[:, None], axis=-1)); mx.eval(s)
            hn += float(s.item()); hc += last - first
        out["held_out_ppl"] = float(np.exp(hn / hc))
    return out


def set_masks(keep):  # keep: bool array (L, NE)
    for i, l in enumerate(layers):
        l.mlp.gate.mask = None if keep[i].all() else mx.array(np.where(keep[i], 0.0, -1e9).astype(np.float32))


rows = []
for frac in (1.0, 0.9, 0.75, 0.6, 0.5, 0.25):
    for scheme in (("per-layer", "global") if frac < 1 else ("none",)):
        keep = np.ones((L, NE), dtype=bool)
        if scheme == "per-layer":
            k = int(round(NE * frac))
            keep[:] = False
            for i in range(L):
                keep[i, np.argsort(-counts[i])[:k]] = True
        elif scheme == "global":
            k = int(round(L * NE * frac))
            thr = np.sort(counts.ravel())[::-1][k - 1]
            keep = counts >= thr
            keep &= np.cumsum(keep.ravel()).reshape(L, NE) <= k
        set_masks(keep)
        r = {"keep_pct": int(frac * 100), "scheme": scheme, "experts_kept": int(keep.sum()),
             "min_kept_in_a_layer": int(keep.sum(1).min()), "calib_activations_covered_pct": round(100 * counts[keep].sum() / counts.sum(), 1)}
        r.update(score())
        print(json.dumps(r), flush=True)
        rows.append(r)
out = {"model": model_dir.name, "calibration": [p.name for p in calib], "calibration_tokens": cal_tokens, "held_out": held and held.name, "rows": rows}
(E.RAW_DIR / f"expert_prune__{model_dir.name}.json").write_text(json.dumps(out, indent=1))
