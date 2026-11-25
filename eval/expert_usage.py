#!/usr/bin/env python3
"""How concentrated is MoE routing? Decides whether keeping only some experts in memory can work.

  expert_usage.py <bf16_model_dir> [text_file ...]      (default text: WikiText-2 test)

Records, for every token and layer, which experts the router picks (mlx-lm, bf16 model), then reports
coverage (how many distinct experts get touched) and simulates an LRU cache over (layer, expert) pairs.
"""
import json, sys
from collections import OrderedDict
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm.models import switch_layers
from mlx_lm.utils import load_model, load_tokenizer

import evallib as E

N_TOKENS, WINDOW = 8192, 2048
model_dir = Path(sys.argv[1]).resolve()
texts = [Path(p) for p in sys.argv[2:]] or [E.WIKI]

model, _ = load_model(model_dir)
tok = load_tokenizer(model_dir)
layers = model.language_model.model.layers if hasattr(model, "language_model") else model.model.layers
ids_of = {id(l.mlp.switch_mlp): i for i, l in enumerate(layers) if hasattr(l.mlp, "switch_mlp")}
n_layers = len(ids_of)
n_exp = layers[0].mlp.switch_mlp.gate_proj.weight.shape[0]
rec = {}
orig = switch_layers.SwitchGLU.__call__


def spy(self, x, indices):
    li = ids_of.get(id(self))
    if li is not None:
        rec.setdefault(li, []).append(np.array(indices).reshape(-1, indices.shape[-1]))
    return orig(self, x, indices)


switch_layers.SwitchGLU.__call__ = spy
out = {"model": model_dir.name, "layers": n_layers, "experts_per_layer": n_exp, "texts": {}}
for path in texts:
    hf = getattr(tok, "_tokenizer", tok)
    ids = list(hf.encode(path.read_text(encoding="utf-8", errors="replace"), add_special_tokens=False))[:N_TOKENS]
    rec.clear()
    for s in range(0, len(ids), WINDOW):
        mx.eval(model(mx.array([ids[s : s + WINDOW]])))
    sel = np.stack([np.concatenate(rec[i]) for i in sorted(rec)])  # (layers, tokens, top_k)
    L, T, K = sel.shape
    r = {"tokens": T, "top_k": K}
    for n in (128, 512, 2048, T):
        distinct = [len(np.unique(sel[l, :n])) for l in range(L)]
        r[f"experts_touched_after_{n}_tokens_pct"] = round(100 * sum(distinct) / (L * n_exp), 1)
    counts = np.stack([np.bincount(sel[l].ravel(), minlength=n_exp) for l in range(L)]).astype(np.float64)
    flat = np.sort(counts.ravel())[::-1]
    cum = np.cumsum(flat) / flat.sum()
    for share in (0.5, 0.8, 0.9, 0.95, 0.99):
        r[f"experts_needed_for_{int(share * 100)}pct_of_activations_pct"] = round(100 * (np.searchsorted(cum, share) + 1) / flat.size, 1)
    # LRU cache over (layer, expert), tokens processed in order as in decoding
    total = L * n_exp
    for cap_pct in (10, 25, 50, 75):
        cap, cache, miss, tok_miss = int(total * cap_pct / 100), OrderedDict(), 0, 0
        warm = T // 4
        for t in range(T):
            m_t = 0
            for l in range(L):
                for e in sel[l, t]:
                    key = l * n_exp + int(e)
                    if key in cache:
                        cache.move_to_end(key)
                    else:
                        m_t += 1
                        cache[key] = 1
                        if len(cache) > cap:
                            cache.popitem(last=False)
            if t >= warm:
                miss += m_t; tok_miss += m_t > 0
        n = (T - warm)
        r[f"lru_{cap_pct}pct"] = {"miss_rate_pct": round(100 * miss / (n * L * K), 2), "expert_loads_per_token": round(miss / n, 2),
                                   "tokens_with_a_miss_pct": round(100 * tok_miss / n, 1)}
    out["texts"][path.name] = r
print(json.dumps(out, indent=1))
(E.RAW_DIR / f"expert_usage__{model_dir.name}.json").write_text(json.dumps(out, indent=1))
