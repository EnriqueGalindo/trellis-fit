#!/usr/bin/env python3
"""KLD of EXL3 quants vs the bf16 model on an arbitrary text (out-of-domain check for calibration effects).

  kld_on_text.py <bf16_dir> <text_file> <exl3_dir> [<exl3_dir> ...]     (PonyExl3 runtime, 8 windows of 2048 tokens)
"""
import json, sys
from pathlib import Path
import mlx.core as mx, mlx.nn as nn, numpy as np
from mlx_lm.utils import load_model, load_tokenizer
from ponyexl3.mlx.model import load_model as load_exl3

bf16, text, quants = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3:]
tok = load_tokenizer(bf16); hf = getattr(tok, "_tokenizer", tok)
ids = list(hf.encode(text.read_text(encoding="utf-8", errors="replace"), add_special_tokens=False))
wins = [ids[i * 2048 : (i + 1) * 2048] for i in range(min(8, len(ids) // 2048))]


def logprobs(model):
    out = []
    for w in wins:
        lp = nn.log_softmax(model(mx.array([w]))[0, 1024:2047, :].astype(mx.float32), axis=-1); mx.eval(lp); out.append(lp)
    return out


ref_model, _ = load_model(bf16); ref = logprobs(ref_model); del ref_model
for q in quants:
    m, _ = load_exl3(q, engine="exl3", warm=True, verbose=False)
    kl, nll, n = 0.0, 0.0, 0
    for w, r, lp in zip(wins, ref, logprobs(m)):
        kl += float(mx.sum(mx.exp(r) * (r - lp)).item())
        nll -= float(mx.sum(mx.take_along_axis(lp, mx.array(w[1025:2048])[:, None], axis=-1)).item()); n += 1023
    print(json.dumps({"quant": Path(q).name, "text": text.name, "kld_mean": round(kl / n, 4), "ppl": round(float(np.exp(nll / n)), 3)}), flush=True)
    del m; mx.clear_cache()
