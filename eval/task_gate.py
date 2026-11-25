#!/usr/bin/env python3
"""Task gate: does a quant still DO things, not just match token distributions?

  task_gate.py <label> <runtime> <model_path> [--tasks humaneval,needles,mmlu] [--limit N]

  runtime: mlxl3 (EXL3 dir) | llama (.gguf) | mlx (bf16 dir, mlx-lm server)

Three checks, all greedy, through each runtime's OpenAI-style chat endpoint (thinking left on, as shipped):
  humaneval  164 Python problems; generated code runs under sandbox-exec (no network, no writes under /Users), 15 s limit
  needles    retrieve 8 planted 6-digit codes from ~6k- and ~14k-token contexts (2 documents each)
  mmlu       150 multiple-choice questions, a fixed sample across all 57 subjects

Results (per item, so two models can be compared item by item) go to results/raw/gate__<label>.json; reruns resume.
"""
from __future__ import annotations

import argparse, gzip, json, os, random, re, signal, socket, subprocess, sys, tempfile, time, urllib.request
from pathlib import Path

import evallib as E

GATE = E.EVAL_DIR / "gate"
LOGS = E.ROOT / "results" / "logs"
CTX = 20480


def free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


class Server:
    def __init__(self, runtime: str, model: str, label: str):
        self.port, self.model = free_port(), model
        tools = E.ROOT / "tools"
        if runtime == "mlxl3":
            eng = Path(os.environ.get("MLXL3_BIN", tools / "mlxl3-engine" / "mlxl3"))
            cmd = [str(eng), "--registry", str(eng.parent / "registry.json"), "serve", model, "--port", str(self.port), "--context-length", str(CTX)]
            self.health, self.model_id = "/ready", model
        elif runtime == "llama":
            d = Path(os.environ.get("LLAMA_DIR", tools / "llama.cpp" / "llama-b12608"))
            cmd = [str(d / "llama-server"), "-m", model, "-c", str(CTX), "-ngl", "999", "-np", "1", "-fit", "off", "--port", str(self.port)]
            self.health, self.model_id = "/health", "gguf"
        elif runtime == "mlx":
            cmd = [sys.executable, "-m", "mlx_lm", "server", "--model", model, "--port", str(self.port)]
            self.health, self.model_id = "/v1/models", model
        else:
            raise SystemExit(f"unknown runtime {runtime}")
        LOGS.mkdir(parents=True, exist_ok=True)
        self.proc = subprocess.Popen(cmd, stdout=open(LOGS / f"gate_server_{label}.log", "w"), stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        t0 = time.time()
        while time.time() - t0 < 900:
            if self.proc.poll() is not None:
                raise RuntimeError(f"server exited {self.proc.returncode}; see gate_server_{label}.log")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{self.port}{self.health}", timeout=3) as r:
                    if r.status == 200:
                        return
            except Exception:
                time.sleep(0.5)
        raise TimeoutError("server not ready")

    def chat(self, prompt: str, max_tokens: int) -> dict:
        body = {"model": self.model_id, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens, "temperature": 0, **EXTRA}
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=900) as r:
                d = json.loads(r.read())
        except Exception as exc:  # keep the run going; the item counts as failed
            return {"answer": "", "error": str(exc)[:200], "seconds": time.time() - t0}
        ch = d["choices"][0]; msg = ch["message"]
        ans = msg.get("content") or ""
        if "</think>" in ans:  # runtimes that leave the reasoning inline
            ans = ans.split("</think>")[-1]
        return {"answer": ans.strip(), "finish": ch.get("finish_reason"), "tokens": (d.get("usage") or {}).get("completion_tokens"), "seconds": time.time() - t0}

    def stop(self):
        if self.proc.poll() is None:
            self.proc.send_signal(signal.SIGTERM)
            try:
                self.proc.wait(30)
            except subprocess.TimeoutExpired:
                self.proc.kill()


EXTRA = json.loads(os.environ.get("GATE_BODY", "{}"))  # e.g. sampling settings merged into every request
TOK = int(os.environ.get("GATE_TOKENS", "0"))  # reply limit override for the first attempt
RETRY = 0  # --retry N: rerun failed items with an N-token reply limit; the first attempt is kept under "first"


def skip(done, key):
    return key in done and not (RETRY and not done[key]["ok"] and "first" not in done[key])


def first(done, key):
    return {"first": {k: v for k, v in done[key].items() if k != "first"}} if key in done else {}


# ------------------------------------------------------------------ humaneval
SANDBOX = '(version 1)(allow default)(deny network*)(deny file-write* (subpath "/Users"))'


def extract_code(ans: str) -> str:
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", ans, flags=re.S)
    return max(blocks, key=len) if blocks else ans


def run_humaneval(srv, done, limit):
    probs = [json.loads(l) for l in gzip.open(GATE / "HumanEval.jsonl.gz", "rt")][:limit]
    for p in probs:
        if skip(done, p["task_id"]):
            continue
        r = srv.chat("Complete this Python function. Reply with the complete function (including its signature and any imports) "
                     "in a single ```python code block and nothing else.\n\n" + p["prompt"], RETRY or TOK or 3072)
        code = extract_code(r["answer"])
        # the prompt goes first so helper functions it defines exist; the model's full definition then replaces the stub
        prog = p["prompt"] + "\n    pass\n\n" + code + "\n\n" + p["test"] + f"\n\ncheck({p['entry_point']})\n"
        passed, why = False, ""
        if code.strip():
            with tempfile.TemporaryDirectory() as td:
                f = Path(td) / "prog.py"; f.write_text(prog)
                try:
                    out = subprocess.run(["/usr/bin/sandbox-exec", "-p", SANDBOX, sys.executable, "-I", str(f)], capture_output=True, text=True, timeout=15, cwd=td)
                    passed, why = out.returncode == 0, out.stderr.strip().splitlines()[-1][:120] if out.stderr.strip() else ""
                except subprocess.TimeoutExpired:
                    why = "timeout"
        else:
            why = "no answer (" + str(r.get("finish") or r.get("error")) + ")"
        done[p["task_id"]] = {"ok": passed, "why": why, "finish": r.get("finish"), "tokens": r.get("tokens"), "seconds": round(r["seconds"], 1),
                              **first(done, p["task_id"])}
        yield


# ------------------------------------------------------------------ needles
NAMES = ["amber", "birch", "cobalt", "delta", "ember", "fjord", "garnet", "harbor"]


def run_needles(srv, done, limit):
    text = E.WIKI.read_text(encoding="utf-8")
    for size_name, chars in (("6k", 25000), ("14k", 58000)):
        for doc in range(2):
            key = f"{size_name}-{doc}"
            if key in done:
                continue
            rnd = random.Random(f"{size_name}{doc}")
            start = rnd.randrange(0, len(text) - chars - 1)
            body = text[start : start + chars]
            codes = {n: f"{rnd.randrange(100000, 999999)}" for n in NAMES}
            paras = body.split("\n")
            slots = sorted(rnd.sample(range(1, len(paras) - 1), len(NAMES)))
            for (n, c), s in zip(codes.items(), reversed(slots)):
                paras.insert(s, f" The secret code for project {n} is {c}. ")
            prompt = ("Read the document, then answer the question after it.\n\n<document>\n" + "\n".join(paras) + "\n</document>\n\n"
                      "The document contains secret codes for these projects: " + ", ".join(NAMES) + ". "
                      "Reply with one line per project in the form `name: code`.")
            r = srv.chat(prompt, 2048)
            found = sum(1 for n, c in codes.items() if re.search(rf"{n}\W+{c}\b", r["answer"]))
            done[key] = {"found": found, "of": len(NAMES), "seconds": round(r["seconds"], 1), "why": r.get("error") or r.get("finish")}
            yield


# ------------------------------------------------------------------ mmlu
def run_mmlu(srv, done, limit):
    qs = json.loads((GATE / "mmlu_150.json").read_text())[:limit]
    for i, q in enumerate(qs):
        key = f"{i:03d}:{q['subject']}"
        if skip(done, key):
            continue
        opts = "\n".join(f"{l}. {c}" for l, c in zip("ABCD", q["choices"]))
        r = srv.chat(f"{q['question']}\n\n{opts}\n\nAnswer with the letter of the correct option only (A, B, C or D).", RETRY or TOK or 4096)
        m = re.findall(r"\b([ABCD])\b", r["answer"])
        pick = m[-1] if m else None
        done[key] = {"ok": pick == "ABCD"[q["answer"]], "pick": pick, "finish": r.get("finish"), "tokens": r.get("tokens"), "seconds": round(r["seconds"], 1),
                     **first(done, key)}
        yield


TASKS = {"humaneval": run_humaneval, "needles": run_needles, "mmlu": run_mmlu}


def summary(doc):
    s = {}
    if doc.get("humaneval"):
        v = doc["humaneval"].values(); s["humaneval"] = f"{sum(x['ok'] for x in v)}/{len(v)}"
    if doc.get("needles"):
        v = doc["needles"].values(); s["needles"] = f"{sum(x['found'] for x in v)}/{sum(x['of'] for x in v)}"
    if doc.get("mmlu"):
        v = doc["mmlu"].values(); s["mmlu"] = f"{sum(x['ok'] for x in v)}/{len(v)}"
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("label"); ap.add_argument("runtime"); ap.add_argument("model")
    ap.add_argument("--tasks", default="needles,humaneval,mmlu"); ap.add_argument("--limit", type=int, default=10**9)
    ap.add_argument("--retry", type=int, default=0)
    a = ap.parse_args()
    global RETRY
    RETRY = a.retry
    out = E.RAW_DIR / f"gate__{a.label}.json"
    doc = json.loads(out.read_text()) if out.exists() else {"label": a.label, "runtime": a.runtime, "model": a.model}
    srv = Server(a.runtime, str(Path(a.model).resolve()), a.label)
    t0 = time.time()
    try:
        for t in a.tasks.split(","):
            done = doc.setdefault(t, {})
            for n, _ in enumerate(TASKS[t](srv, done, a.limit)):
                if n % 10 == 0:
                    out.write_text(json.dumps(doc, indent=1))
            out.write_text(json.dumps(doc, indent=1))
            print(json.dumps({"label": a.label, "after": t, **summary(doc), "minutes": round((time.time() - t0) / 60, 1)}), flush=True)
    finally:
        srv.stop()
        out.write_text(json.dumps(doc, indent=1))


if __name__ == "__main__":
    main()
