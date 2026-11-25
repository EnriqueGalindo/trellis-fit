"""In-process tests for the two Python runtimes.

  runtime `pony` : EXL3 checkpoint through PonyExl3
  runtime `mlx`  : plain mlx-lm model (the bf16 source; this is the KLD reference)

Run with the PonyExl3 venv python. One test per process so memory numbers are clean:

  python worker_mlx.py <load|ppl|speed|mem> <model_dir> <pony|mlx> [--ref NAME] [--save-ref NAME]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

import evallib as E


def load(model_dir: str, runtime: str):
    from mlx_lm.utils import load_tokenizer

    t0 = time.perf_counter()
    if runtime == "pony":
        from ponyexl3.mlx.model import load_model

        model, config = load_model(model_dir, engine="exl3", warm=True, verbose=False)
    elif runtime == "mlx":
        from mlx_lm.utils import load_model

        model, config = load_model(Path(model_dir))
        mx.eval(model.parameters())
    else:
        raise SystemExit(f"unknown runtime {runtime}")
    if os.environ.get("EVAL_CLEAR_CACHE_AFTER_LOAD") == "1":
        # PonyExl3 leaves its load-time scratch buffers in MLX's reuse cache (54 GiB on the
        # 122B). Opt-in: hand them back before measuring, as an application could.
        mx.clear_cache()
    tok = load_tokenizer(Path(model_dir))
    return model, config, tok, time.perf_counter() - t0


def stream(model, runtime: str, prompt_ids: list[int], max_tokens: int):
    """Plain greedy decode with EOS disabled, so every run makes max_tokens.

    PonyExl3's loop accepts bare mlx-lm models too, so the bf16 reference and the
    EXL3 quants are timed by the same code (no speculative drafter, no MTP)."""
    from ponyexl3.mlx.generate import GenStats, stream_generate

    stats = GenStats()
    gen = stream_generate(
        model, prompt_ids, max_tokens=max_tokens, temp=0.0, eos_ids=frozenset(), stats=stats
    )
    return gen, stats


def timed_generate(model, runtime, prompt_ids, max_tokens):
    t0 = time.perf_counter()
    gen, stats = stream(model, runtime, prompt_ids, max_tokens)
    toks, ttft = [], None
    for t in gen:
        if ttft is None:
            ttft = time.perf_counter() - t0
        toks.append(t)
    return {
        "prompt_tokens": stats.prompt_tokens,
        "gen_tokens": stats.gen_tokens,
        "prefill_tps": stats.prompt_tokens / stats.prefill_s,
        "decode_tps": stats.gen_tokens / stats.decode_s if stats.decode_s else 0.0,
        "ttft_s": ttft,
        "total_s": time.perf_counter() - t0,
    }, toks


# ------------------------------------------------------------------ tests
def t_load(a):
    model, config, tok, load_s = load(a.model, a.runtime)
    ids = list(tok.encode(E.LOAD_PROMPT))
    _, out = timed_generate(model, a.runtime, ids, 24)
    text = tok.decode(out)
    sane = "Paris" in text
    res = {
        "ok": True,
        "sane": sane,
        "load_s": load_s,
        "model_type": config.get("model_type"),
        "prompt": E.LOAD_PROMPT,
        "completion": text,
    }
    print(json.dumps(res, ensure_ascii=False))
    E.save_section(a.model, a.runtime, "load", res)


def t_ppl(a):
    model, _, tok, _ = load(a.model, a.runtime)
    ids, bos = E.wiki_tokens(tok)
    wins = E.windows(ids, bos)
    sha = E.tokens_sha(wins)
    first, last = E.scored_range()

    ref_dir = E.REF_DIR / a.ref if a.ref else None
    save_dir = E.REF_DIR / a.save_ref if a.save_ref else None
    if ref_dir:
        meta = json.loads((ref_dir / "meta.json").read_text())
        if meta["tokens_sha"] != sha:
            raise SystemExit(
                f"reference {a.ref} was built on a different token slice "
                f"({meta['tokens_sha']} vs {sha}); KLD would be meaningless"
            )
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    nll_sum, n, top1, klds, ref_top1_agree = 0.0, 0, 0, [], 0
    t0 = time.perf_counter()
    for i, w in enumerate(wins):
        x = mx.array([w])
        logits = model(x)[0, first:last, :].astype(mx.float32)
        lp = nn.log_softmax(logits, axis=-1)
        tgt = mx.array(w[first + 1 : last + 1])
        nll = -mx.take_along_axis(lp, tgt[:, None], axis=-1)[:, 0]
        pred = mx.argmax(lp, axis=-1)
        mx.eval(lp, nll, pred)
        nll_sum += float(mx.sum(nll).item())
        top1 += int(mx.sum(pred == tgt).item())
        n += int(tgt.shape[0])
        if save_dir:
            np.save(save_dir / f"{i:02d}.npy", np.array(lp.astype(mx.float16)))
        if ref_dir:
            ref = mx.array(np.load(ref_dir / f"{i:02d}.npy")).astype(mx.float32)
            # KL(ref || model), per token, summed over the vocabulary
            kl = mx.sum(mx.exp(ref) * (ref - lp), axis=-1)
            agree = mx.sum(mx.argmax(ref, axis=-1) == pred)
            mx.eval(kl, agree)
            klds.extend(float(v) for v in kl.tolist())
            ref_top1_agree += int(agree.item())
            del ref, kl
        del logits, lp, nll, pred
        mx.clear_cache()
        print(f"  window {i + 1}/{len(wins)} ppl so far {np.exp(nll_sum / n):.4f}", file=sys.stderr)

    res = E.summarize_nll(nll_sum, n)
    res.update(
        {
            "top1_acc": top1 / n,
            "n_ctx": E.N_CTX,
            "n_chunks": E.N_CHUNKS,
            "tokens_sha": sha,
            "seconds": time.perf_counter() - t0,
        }
    )
    if save_dir:
        (save_dir / "meta.json").write_text(
            json.dumps({"tokens_sha": sha, "model": a.model, "ppl": res["ppl"], "top1_acc": res["top1_acc"]})
        )
        (save_dir / "windows.json").write_text(json.dumps(wins))
        res["saved_ref"] = a.save_ref
    print(json.dumps(res))
    E.save_section(a.model, a.runtime, "ppl", res)
    if ref_dir:
        k = E.summarize_kld(klds)
        k.update({"ref": a.ref, "ref_ppl": meta["ppl"], "top1_agree_with_ref": ref_top1_agree / n})
        print(json.dumps(k))
        E.save_section(a.model, a.runtime, "kld", k)


def t_speed(a):
    model, _, tok, _ = load(a.model, a.runtime)
    ids, _ = E.wiki_tokens(tok)
    prompt = ids[: E.SPEED_PROMPT]
    timed_generate(model, a.runtime, prompt, 16)  # warmup, discarded
    runs = []
    for _ in range(E.SPEED_RUNS):
        r, _ = timed_generate(model, a.runtime, prompt, E.SPEED_GEN)
        runs.append(r)
        mx.clear_cache()
    res = {
        "decode_tps": E.median([r["decode_tps"] for r in runs]),
        "prefill_tps": E.median([r["prefill_tps"] for r in runs]),
        "ttft_s": E.median([r["ttft_s"] for r in runs]),
        "prompt_tokens": E.SPEED_PROMPT,
        "gen_tokens": runs[0]["gen_tokens"],
        "runs": runs,
    }
    print(json.dumps(res))
    E.save_section(a.model, a.runtime, "speed", res)


def t_mem(a):
    model, _, tok, _ = load(a.model, a.runtime)
    after_load = E.footprint()
    mx.reset_peak_memory()
    ids, _ = E.wiki_tokens(tok)
    prompt = ids[: E.MEM_CTX - E.SPEED_GEN]
    r, _ = timed_generate(model, a.runtime, prompt, E.SPEED_GEN)
    end = E.footprint()
    res = {
        "context_tokens": r["prompt_tokens"] + r["gen_tokens"],
        "loaded_footprint_gib": after_load["footprint_gib"],
        "load_peak_footprint_gib": after_load["peak_footprint_gib"],
        "peak_footprint_gib": end["peak_footprint_gib"],
        "mlx_peak_gib_during_8k": mx.get_peak_memory() / E.GIB,
        "mlx_active_gib": mx.get_active_memory() / E.GIB,
        "prefill_tps_8k": r["prefill_tps"],
        "decode_tps_8k": r["decode_tps"],
    }
    print(json.dumps(res))
    E.save_section(a.model, a.runtime, "mem", res)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("test", choices=["load", "ppl", "speed", "mem"])
    ap.add_argument("model")
    ap.add_argument("runtime", choices=["pony", "mlx"])
    ap.add_argument("--ref", default=None, help="reference name under eval/refs for KLD")
    ap.add_argument("--save-ref", default=None, help="save this model's log-probs as a reference")
    a = ap.parse_args()
    a.model = str(Path(a.model).resolve())
    {"load": t_load, "ppl": t_ppl, "speed": t_speed, "mem": t_mem}[a.test](a)
