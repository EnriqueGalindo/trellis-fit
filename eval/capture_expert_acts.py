#!/usr/bin/env python3
"""Capture calibration inputs for routed MoE experts, which PonyExl3's own capture cannot see.

  capture_expert_acts.py <bf16_dir> <text> <out.npz> <layer> <rows> <expert> [<expert> ...]

For each chosen expert of the layer: gate_proj/up_proj inputs are the hidden states of tokens routed to it;
down_proj inputs are silu(gate(x)) * up(x) for those tokens. Output keys are EXL3 module names, the format
`ponyexl3-convert-advanced --calibration-activations-map` reads.
"""
import sys
from pathlib import Path
import mlx.core as mx, mlx.nn as nn, numpy as np
from mlx_lm.models import switch_layers
from mlx_lm.utils import load_model, load_tokenizer

bf16, text, out, li, rows = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
experts = [int(e) for e in sys.argv[6:]]
model, _ = load_model(bf16); tok = load_tokenizer(bf16); hf = getattr(tok, "_tokenizer", tok)
layer = model.language_model.model.layers[li]; sw = layer.mlp.switch_mlp
Wg, Wu = sw.gate_proj.weight, sw.up_proj.weight  # (E, hidden, in)
acc = {e: {"x": [], "h": [], "n": 0} for e in experts}
orig = switch_layers.SwitchGLU.__call__


def spy(self, x, indices):
    if self is sw:
        xf = x.reshape(-1, x.shape[-1]); idx = np.array(indices).reshape(-1, indices.shape[-1])
        for e in experts:
            if acc[e]["n"] >= rows:
                continue
            sel = np.nonzero((idx == e).any(axis=1))[0]
            if sel.size == 0:
                continue
            xe = xf[mx.array(sel)].astype(mx.float32)
            h = nn.silu(xe @ Wg[e].astype(mx.float32).T) * (xe @ Wu[e].astype(mx.float32).T)
            acc[e]["x"].append(np.array(xe.astype(mx.float16))); acc[e]["h"].append(np.array(h.astype(mx.float16))); acc[e]["n"] += sel.size
    return orig(self, x, indices)


switch_layers.SwitchGLU.__call__ = spy
ids = list(hf.encode(text.read_text(encoding="utf-8", errors="replace"), add_special_tokens=False))
used = 0
for s in range(0, len(ids) - 2048, 2048):
    mx.eval(model(mx.array([ids[s : s + 2048]]))); used += 2048
    if all(a["n"] >= rows for a in acc.values()):
        break
pre = f"model.language_model.layers.{li}.mlp.experts"
data = {}
for e, a in acc.items():
    x, h = np.concatenate(a["x"])[:rows], np.concatenate(a["h"])[:rows]
    data[f"{pre}.{e}.gate_proj"] = x; data[f"{pre}.{e}.up_proj"] = x; data[f"{pre}.{e}.down_proj"] = h
np.savez(out, **data)
print({"tokens_run": used, "rows_per_expert": {e: min(a["n"], rows) for e, a in acc.items()}})
