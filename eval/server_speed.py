#!/usr/bin/env python3
"""Generation speed through the MLXL3 server: server_speed.py <engine_dir> <model_dir>...
Rate = extra tokens / extra seconds between an 8-token and a 1,200-token reply to the same prompt."""
import json, subprocess, sys, time, urllib.request
from pathlib import Path
eng = Path(sys.argv[1]).resolve()
import evallib as E
text = E.WIKI.read_text(encoding="utf-8")[:30000]
def call(port, model, prompt, n):
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": n, "temperature": 0}).encode()
    t = time.time()
    r = json.load(urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", body, {"Content-Type": "application/json"}), timeout=900))
    return time.time() - t, r["usage"]
for i, m in enumerate(sys.argv[2:]):
    m = str(Path(m).resolve()); port = 8770 + i
    p = subprocess.Popen([str(eng / "mlxl3"), "--registry", str(eng / "registry.json"), "serve", m, "--port", str(port), "--context-length", "65536"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(300):
            try: urllib.request.urlopen(f"http://127.0.0.1:{port}/ready", timeout=2); break
            except Exception: time.sleep(1)
        for name, prompt in [("short", "Write a detailed 2000-word essay on the history of bridges."), ("8k ctx", text + "\n\nWrite a detailed 2000-word essay discussing the text above.")]:
            call(port, m, prompt, 8)
            for rep in range(2):
                ta, ua = call(port, m, prompt, 8); tb, ub = call(port, m, prompt, 1200)
                print(json.dumps({"model": Path(m).name, "ctx": name, "prompt_tokens": ub["prompt_tokens"], "out": ub["completion_tokens"], "tok_s": round((ub["completion_tokens"] - ua["completion_tokens"]) / (tb - ta), 1), "t8": round(ta, 2), "t1200": round(tb, 2)}), flush=True)
    finally:
        p.terminate(); p.wait(); subprocess.run(["pkill", "-f", "mlxl3-fork-engine-aw/mlxl3"]); time.sleep(8)
