"""Tests for GGUF models through the prebuilt llama.cpp binaries.

  python worker_llama.py <load|ppl|speed|mem> <model.gguf> [--ref NAME]

PPL comes from `llama-perplexity` itself (-c N_CTX --chunks N_CHUNKS), which is the
convention the MLX workers copy. KLD is computed here, against the same MLX bf16
reference the EXL3 quants use: llama-perplexity dumps its per-token log-probs
(--save-all-logits) and this script compares them position by position.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import socket
import struct
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import numpy as np

import evallib as E

LLAMA_DIR = Path(os.environ.get("LLAMA_DIR", E.ROOT / "tools" / "llama.cpp" / "llama-b12359"))
LOG_DIR = E.ROOT / "results" / "logs"


# ------------------------------------------------------------------ server
class Server:
    def __init__(self, gguf: str, ctx: int, mmap: bool = True):
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        self.port = s.getsockname()[1]
        s.close()
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        self.log = LOG_DIR / f"llama-server_{Path(gguf).stem}.log"
        cmd = [
            str(LLAMA_DIR / "llama-server"), "-m", gguf, "-c", str(ctx), "-ngl", "999",
            "-np", "1", "-fit", "off", "-b", "2048", "-ub", "2048",
            "--host", "127.0.0.1", "--port", str(self.port),
        ]
        if not mmap:
            cmd += ["--load-mode", "none"]  # this build replaced --no-mmap
        cmd += os.environ.get("LLAMA_EXTRA_ARGS", "").split()  # e.g. "--cpu-moe --moe-cache-mib 4096"
        self.cmd = cmd
        self.t0 = time.perf_counter()
        self.proc = subprocess.Popen(cmd, stdout=open(self.log, "w"), stderr=subprocess.STDOUT)
        self.load_s = None

    def wait(self, timeout=3600):
        while time.perf_counter() - self.t0 < timeout:
            if self.proc.poll() is not None:
                tail = self.log.read_text(errors="replace")[-3000:]
                raise RuntimeError(f"llama-server exited {self.proc.returncode}:\n{tail}")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=2) as r:
                    if r.status == 200:
                        self.load_s = time.perf_counter() - self.t0
                        return self
            except Exception:
                time.sleep(0.25)
        raise TimeoutError("llama-server did not become healthy")

    def post(self, path: str, body: dict, timeout=3600):
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}{path}",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        return urllib.request.urlopen(req, timeout=timeout)

    def tokenize(self, text: str) -> list[int]:
        with self.post("/tokenize", {"content": text, "add_special": True}) as r:
            return json.loads(r.read())["tokens"]

    def generate(self, prompt, n: int) -> dict:
        """Greedy, EOS ignored, no prompt cache. Returns llama.cpp's own timings + client TTFT."""
        body = {
            "prompt": prompt, "n_predict": n, "temperature": 0.0, "top_k": 1, "seed": 0,
            "cache_prompt": False, "ignore_eos": True, "stream": True,
        }
        t0 = time.perf_counter()
        ttft, text, last = None, [], {}
        with self.post("/completion", body) as r:
            for line in r:
                line = line.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                ev = json.loads(line[5:])
                if ev.get("content"):
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    text.append(ev["content"])
                if ev.get("stop"):
                    last = ev
        tm = last.get("timings", {})
        return {
            "text": "".join(text),
            "prompt_tokens": tm.get("prompt_n"),
            "gen_tokens": tm.get("predicted_n"),
            "prefill_tps": tm.get("prompt_per_second"),
            "decode_tps": tm.get("predicted_per_second"),
            "ttft_s": ttft,
            "total_s": time.perf_counter() - t0,
        }

    def stop(self):
        if self.proc.poll() is None:
            self.proc.send_signal(signal.SIGTERM)
            try:
                self.proc.wait(30)
            except subprocess.TimeoutExpired:
                self.proc.kill()


def wiki_prefix_tokens(srv: Server, n: int) -> list[int]:
    # tokenizing a generous prefix is enough; ~3 chars/token is a safe floor
    text = E.WIKI.read_text(encoding="utf-8")[: n * 12]
    toks = srv.tokenize(text)
    if len(toks) < n:
        toks = srv.tokenize(E.WIKI.read_text(encoding="utf-8"))
    return toks[:n]


# ------------------------------------------------------------------ tests
def t_load(a):
    srv = Server(a.model, 2048)
    try:
        srv.wait()
        r = srv.generate(E.LOAD_PROMPT, 24)
    finally:
        srv.stop()
    res = {
        "ok": True, "sane": "Paris" in r["text"], "load_s": srv.load_s,
        "prompt": E.LOAD_PROMPT, "completion": r["text"],
    }
    print(json.dumps(res, ensure_ascii=False))
    E.save_section(a.model, "llama", "load", res)


def read_logits_file(path: Path):
    """Parse llama-perplexity's --save-all-logits dump. Yields (chunk_index, log_probs[float32])."""
    f = open(path, "rb")
    if f.read(8) != b"_logits_":
        raise ValueError("not a llama-perplexity logits file")
    n_ctx, n_vocab, n_chunk = struct.unpack("<iii", f.read(12))
    tokens = np.frombuffer(f.read(4 * n_ctx * n_chunk), dtype=np.int32).reshape(n_chunk, n_ctx)
    nv = 2 * ((n_vocab + 1) // 2) + 4
    n_scored = n_ctx - 1 - n_ctx // 2

    def chunks():
        for i in range(n_chunk):
            raw = np.frombuffer(f.read(2 * nv * n_scored), dtype=np.uint16).reshape(n_scored, nv)
            hdr = raw[:, :4].copy().view(np.float32)  # [scale, min_log_prob]
            q = raw[:, 4 : 4 + n_vocab].astype(np.float32)
            yield i, q * hdr[:, 0:1] + hdr[:, 1:2]

    return n_ctx, n_vocab, n_chunk, tokens, chunks()


def t_ppl(a):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log = LOG_DIR / f"llama-perplexity_{Path(a.model).stem}.log"
    tmp = Path(tempfile.mkdtemp(prefix="llama_logits_", dir=E.EVAL_DIR)) / "logits.bin"
    cmd = [
        str(LLAMA_DIR / "llama-perplexity"), "-m", a.model, "-f", str(E.WIKI),
        "-c", str(E.N_CTX), "--chunks", str(E.N_CHUNKS), "-b", str(E.N_CTX),
        "-ngl", "999", "--save-all-logits", str(tmp),
    ]
    t0 = time.perf_counter()
    with open(log, "w") as lf:
        rc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT).returncode
    out = log.read_text(errors="replace")
    if rc != 0:
        raise SystemExit(f"llama-perplexity exited {rc}:\n{out[-3000:]}")
    m = re.search(r"Final estimate: PPL = ([0-9.]+) \+/- ([0-9.]+)", out)
    if not m:
        raise SystemExit(f"no PPL in llama-perplexity output:\n{out[-2000:]}")
    res = {
        "ppl": float(m.group(1)), "ppl_stderr": float(m.group(2)),
        "n_ctx": E.N_CTX, "n_chunks": E.N_CHUNKS,
        "scored_tokens": E.N_CHUNKS * (E.N_CTX - 1 - E.N_CTX // 2),
        "seconds": time.perf_counter() - t0,
    }

    n_ctx, n_vocab, n_chunk, tokens, chunks = read_logits_file(tmp)
    first, last = E.scored_range(n_ctx)
    ref_dir = E.REF_DIR / a.ref if a.ref else None
    ref_wins = json.loads((ref_dir / "windows.json").read_text()) if ref_dir else None
    same_tokens = None
    if ref_wins:
        # llama.cpp scores with BOS forced into position 0; only positions >= first matter
        same_tokens = all(
            list(tokens[i, first : last + 1]) == ref_wins[i][first : last + 1] for i in range(n_chunk)
        ) and n_chunk == len(ref_wins)
        res["tokens_match_reference"] = bool(same_tokens)
    nll, n, top1, klds, agree = 0.0, 0, 0, [], 0
    for i, lp in chunks:
        tgt = tokens[i, first + 1 : last + 1]
        nll -= float(lp[np.arange(len(tgt)), tgt].sum())
        pred = lp.argmax(-1)
        top1 += int((pred == tgt).sum())
        n += len(tgt)
        if ref_dir and same_tokens:
            ref = np.load(ref_dir / f"{i:02d}.npy").astype(np.float32)
            if ref.shape != lp.shape:
                raise SystemExit(f"vocab mismatch vs reference: {ref.shape} vs {lp.shape}")
            klds.extend((np.exp(ref) * (ref - lp)).sum(-1).tolist())
            agree += int((ref.argmax(-1) == pred).sum())
    res["ppl_from_dump"] = float(np.exp(nll / n))
    res["top1_acc"] = top1 / n
    tmp.unlink()
    tmp.parent.rmdir()
    print(json.dumps(res))
    E.save_section(a.model, "llama", "ppl", res)
    if ref_dir and not same_tokens:
        print("KLD skipped: GGUF tokenization differs from the reference slice", file=sys.stderr)
    if klds:
        meta = json.loads((ref_dir / "meta.json").read_text())
        k = E.summarize_kld(klds)
        k.update({"ref": a.ref, "ref_ppl": meta["ppl"], "top1_agree_with_ref": agree / n})
        print(json.dumps(k))
        E.save_section(a.model, "llama", "kld", k)


def t_speed(a):
    srv = Server(a.model, 2048)
    try:
        srv.wait()
        prompt = wiki_prefix_tokens(srv, E.SPEED_PROMPT)
        srv.generate(prompt, 16)  # warmup, discarded
        runs = [srv.generate(prompt, E.SPEED_GEN) for _ in range(E.SPEED_RUNS)]
    finally:
        srv.stop()
    for r in runs:
        r.pop("text")
    res = {
        "decode_tps": E.median([r["decode_tps"] for r in runs]),
        "prefill_tps": E.median([r["prefill_tps"] for r in runs]),
        "ttft_s": E.median([r["ttft_s"] for r in runs]),
        "prompt_tokens": runs[0]["prompt_tokens"], "gen_tokens": runs[0]["gen_tokens"],
        "runs": runs,
    }
    print(json.dumps(res))
    E.save_section(a.model, "llama", "speed", res)


def t_mem(a):
    # --no-mmap: with mmap the weights are file-backed pages that the footprint
    # counter does not see, which would make GGUF look almost free.
    srv = Server(a.model, E.MEM_CTX, mmap=False)
    try:
        srv.wait()
        loaded = E.footprint(srv.proc.pid)
        prompt = wiki_prefix_tokens(srv, E.MEM_CTX - E.SPEED_GEN)
        r = srv.generate(prompt, E.SPEED_GEN)
        end = E.footprint(srv.proc.pid)
    finally:
        srv.stop()
    res = {
        "context_tokens": (r["prompt_tokens"] or 0) + (r["gen_tokens"] or 0),
        "loaded_footprint_gib": loaded["footprint_gib"],
        "load_peak_footprint_gib": loaded["peak_footprint_gib"],
        "peak_footprint_gib": end["peak_footprint_gib"],
        "prefill_tps_8k": r["prefill_tps"], "decode_tps_8k": r["decode_tps"],
        "mmap": False,
    }
    print(json.dumps(res))
    E.save_section(a.model, "llama", "mem", res)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("test", choices=["load", "ppl", "speed", "mem"])
    ap.add_argument("model")
    ap.add_argument("--ref", default=None)
    a = ap.parse_args()
    a.model = str(Path(a.model).resolve())
    {"load": t_load, "ppl": t_ppl, "speed": t_speed, "mem": t_mem}[a.test](a)
