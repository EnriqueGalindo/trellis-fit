"""Tests for EXL3 checkpoints through the MLXL3 engine binary (no Desktop app needed).

  python worker_mlxl3.py <load|ppl|speed|mem> <model_dir> [--ref NAME]

What the engine CLI allows, and so what this measures:

* `mlxl3 forward --tokens ...` prints raw fp16 logits per position. PPL and KLD are
  computed from those on the exact token windows the other runtimes use.
* `mlxl3 run --prompt ...` is greedy, always applies the chat template and always stops
  at EOS; it prints its own `[prefill .. tok/s · decode .. tok/s · TTFT .. ms · N tokens]`
  line. Speed therefore uses the same 512 WikiText tokens wrapped in a chat turn (a few
  template tokens more than the raw prompt elsewhere) and can end before 128 tokens.
  The model is reloaded for each of the 3 runs; load time is not part of TTFT.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

import evallib as E

ENGINE = Path(os.environ.get("MLXL3_BIN", E.ROOT / "tools" / "mlxl3-engine" / "mlxl3"))
REGISTRY = ENGINE.parent / "registry.json"
STATS = re.compile(r"\[prefill ([0-9.]+) tok/s · decode ([0-9.]+) tok/s · TTFT ([0-9.]+) ms · (\d+) tokens\]")


def base(*args: str) -> list[str]:
    return [str(ENGINE), "--registry", str(REGISTRY), *args]


def tokenizer(model: str):
    from mlx_lm.utils import load_tokenizer

    return load_tokenizer(Path(model))


def run_chat(model: str, prompt: str, max_tokens: int, watch_mem: bool = False) -> dict:
    t0 = time.perf_counter()
    p = subprocess.Popen(
        base("run", model, "--prompt", prompt, "--max-tokens", str(max_tokens)),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.DEVNULL, text=True,
    )
    peak = None
    if watch_mem:
        while p.poll() is None:
            try:
                peak = E.footprint(p.pid)
            except OSError:
                pass
            time.sleep(0.05)
    out, err = p.communicate()
    if p.returncode != 0:
        raise RuntimeError(f"mlxl3 run exited {p.returncode}: {(err or out).strip()[-2000:]}")
    m = STATS.search(out)
    res = {"text": STATS.sub("", out).strip(), "wall_s": time.perf_counter() - t0}
    if m:
        res.update(prefill_tps=float(m.group(1)), decode_tps=float(m.group(2)),
                   ttft_s=float(m.group(3)) / 1000, gen_tokens=int(m.group(4)))
    if peak:
        res["peak"] = peak
    return res


def t_load(a):
    t0 = time.perf_counter()
    r = run_chat(a.model, "What is the capital of France? Answer with just the city name.", 256)
    res = {
        "ok": True, "sane": "Paris" in r["text"], "wall_s_including_load": time.perf_counter() - t0,
        "prompt": "What is the capital of France? Answer with just the city name. (chat turn)",
        "completion": r["text"][-400:],
    }
    print(json.dumps(res, ensure_ascii=False))
    E.save_section(a.model, "mlxl3", "load", res)


def forward_logprobs(model: str, window: list[int], first: int, last: int) -> np.ndarray:
    """log-softmax over the vocabulary for positions [first, last) of one window."""
    p = subprocess.Popen(
        base("forward", model, "--tokens", ",".join(map(str, window[:last]))),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.DEVNULL,
    )
    rows = []
    for i, line in enumerate(p.stdout):
        if i < first:
            continue  # unscored context positions: skip the (large) JSON parse
        row = json.loads(line)
        if row["token"] != window[i]:
            raise RuntimeError(f"forward row {i} echoes token {row['token']}, sent {window[i]}")
        x = np.array(row["logits"], dtype=np.uint16).view(np.float16).astype(np.float32)
        x -= x.max()
        rows.append(x - np.log(np.exp(x).sum()))
    err = p.stderr.read().decode(errors="replace")
    if p.wait() != 0 or len(rows) != last - first:
        raise RuntimeError(f"mlxl3 forward exited {p.returncode} with {len(rows)} rows: {err.strip()[-2000:]}")
    return np.stack(rows)


def t_ppl(a):
    tok = tokenizer(a.model)
    ids, bos = E.wiki_tokens(tok)
    wins = E.windows(ids, bos)
    sha = E.tokens_sha(wins)
    first, last = E.scored_range()
    ref_dir = E.REF_DIR / a.ref if a.ref else None
    if ref_dir:
        meta = json.loads((ref_dir / "meta.json").read_text())
        if meta["tokens_sha"] != sha:
            raise SystemExit(f"reference {a.ref} was built on a different token slice")
    nll, n, top1, klds, agree = 0.0, 0, 0, [], 0
    t0 = time.perf_counter()
    for i, w in enumerate(wins):
        lp = forward_logprobs(a.model, w, first, last)
        tgt = np.array(w[first + 1 : last + 1])
        nll -= float(lp[np.arange(len(tgt)), tgt].sum())
        pred = lp.argmax(-1)
        top1 += int((pred == tgt).sum())
        n += len(tgt)
        if ref_dir:
            ref = np.load(ref_dir / f"{i:02d}.npy").astype(np.float32)
            klds.extend((np.exp(ref) * (ref - lp)).sum(-1).tolist())
            agree += int((ref.argmax(-1) == pred).sum())
        print(f"  window {i + 1}/{len(wins)} ppl so far {np.exp(nll / n):.4f}", file=sys.stderr)
    res = E.summarize_nll(nll, n)
    res.update({"top1_acc": top1 / n, "n_ctx": E.N_CTX, "n_chunks": E.N_CHUNKS, "tokens_sha": sha,
                "seconds": time.perf_counter() - t0})
    print(json.dumps(res))
    E.save_section(a.model, "mlxl3", "ppl", res)
    if klds:
        k = E.summarize_kld(klds)
        k.update({"ref": a.ref, "ref_ppl": meta["ppl"], "top1_agree_with_ref": agree / n})
        print(json.dumps(k))
        E.save_section(a.model, "mlxl3", "kld", k)


def wiki_prompt(model: str, n_tokens: int) -> str:
    tok = tokenizer(model)
    ids, _ = E.wiki_tokens(tok)
    return tok.decode(ids[:n_tokens])


def t_speed(a):
    prompt = wiki_prompt(a.model, E.SPEED_PROMPT)
    run_chat(a.model, prompt, 16)  # warmup (page cache), discarded
    runs = []
    for _ in range(E.SPEED_RUNS):
        r = run_chat(a.model, prompt, E.SPEED_GEN)
        r.pop("text")
        runs.append(r)
    if "decode_tps" not in runs[0]:
        raise SystemExit("mlxl3 run printed no stats line")
    res = {
        "decode_tps": E.median([r["decode_tps"] for r in runs]),
        "prefill_tps": E.median([r["prefill_tps"] for r in runs]),
        "ttft_s": E.median([r["ttft_s"] for r in runs]),
        "prompt_tokens": f"{E.SPEED_PROMPT} + chat template",
        "gen_tokens": runs[0]["gen_tokens"],
        "runs": runs,
    }
    print(json.dumps(res))
    E.save_section(a.model, "mlxl3", "speed", res)


def t_mem(a):
    prompt = wiki_prompt(a.model, E.MEM_CTX - E.SPEED_GEN - 32)  # leave room for template tokens
    r = run_chat(a.model, prompt, E.SPEED_GEN, watch_mem=True)
    res = {
        "context_tokens": f"~{E.MEM_CTX} (prompt {E.MEM_CTX - E.SPEED_GEN - 32} + template + {r.get('gen_tokens')} generated)",
        "peak_footprint_gib": r["peak"]["peak_footprint_gib"],
        "last_footprint_gib": r["peak"]["footprint_gib"],
        "prefill_tps_8k": r.get("prefill_tps"), "decode_tps_8k": r.get("decode_tps"),
    }
    print(json.dumps(res))
    E.save_section(a.model, "mlxl3", "mem", res)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("test", choices=["load", "ppl", "speed", "mem"])
    ap.add_argument("model")
    ap.add_argument("--ref", default=None)
    a = ap.parse_args()
    a.model = str(Path(a.model).resolve())
    {"load": t_load, "ppl": t_ppl, "speed": t_speed, "mem": t_mem}[a.test](a)
