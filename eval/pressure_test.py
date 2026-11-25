#!/usr/bin/env python3
"""Rerun candidates with only <A> GiB of memory left for them, to imitate a 16 GB Mac.

  pressure_test.py <available_gib> [<available_gib> ...]

"Available" = free + file-cache pages after the ballast is pinned, i.e. what a model process can claim
before macOS must compress or swap someone else. A clean 16 GB mini has roughly 12 to 13 GiB of that.
Not imitated: the mini's lower GPU memory cap and its slower chip and SSD.
"""
import json, os, re, subprocess, sys, time
from pathlib import Path
import evallib as E

HERE = Path(__file__).resolve().parent
M = E.ROOT / "models"
PAGE = 16384


def vm():
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    g = lambda k: int(re.search(rf"{k}:\s+(\d+)", out).group(1)) * PAGE / E.GIB
    swap = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout
    return {"free": g("Pages free"), "file": g("File-backed pages"), "swap_used_gib": float(re.search(r"used = ([0-9.]+)M", swap).group(1)) / 1024}


def avail():
    v = vm(); return v["free"] + v["file"]


def llama(gguf, build):
    os.environ["LLAMA_DIR"] = str(E.ROOT / "tools" / "llama.cpp" / build)
    os.environ["LLAMA_EXTRA_ARGS"] = ""
    import importlib, worker_llama as W
    importlib.reload(W)
    srv = W.Server(str(gguf), E.MEM_CTX)  # default: mmap
    try:
        srv.wait(timeout=600)
        p = W.wiki_prefix_tokens(srv, E.SPEED_PROMPT)
        srv.generate(p, 16)
        runs = [srv.generate(p, E.SPEED_GEN) for _ in range(3)]
        r8 = srv.generate(W.wiki_prefix_tokens(srv, E.MEM_CTX - E.SPEED_GEN), E.SPEED_GEN)
        f = E.footprint(srv.proc.pid)
        return {"ok": True, "load_s": round(srv.load_s, 1), "decode_tps": round(E.median([r["decode_tps"] for r in runs]), 1),
                "prefill_tps": round(E.median([r["prefill_tps"] for r in runs])), "decode_tps_8k": round(r8["decode_tps"], 1),
                "prefill_tps_8k": round(r8["prefill_tps"]), "must_keep_gib": round(f["peak_footprint_gib"], 2),
                "resident_gib": round(f["rss_gib"], 2), "pageins": f["pageins"]}
    except Exception as exc:
        return {"ok": False, "error": str(exc)[-300:]}
    finally:
        srv.stop()


def mlxl3(model):
    import worker_mlxl3 as X
    try:
        t0 = time.perf_counter()
        prompt = X.wiki_prompt(str(model), E.SPEED_PROMPT)
        runs = [X.run_chat(str(model), prompt, E.SPEED_GEN, watch_mem=True) for _ in range(3)]
        r8 = X.run_chat(str(model), X.wiki_prompt(str(model), E.MEM_CTX - E.SPEED_GEN - 32), E.SPEED_GEN, watch_mem=True)
        return {"ok": True, "wall_s_4_runs": round(time.perf_counter() - t0, 1), "decode_tps": round(E.median([r["decode_tps"] for r in runs]), 1),
                "prefill_tps": round(E.median([r["prefill_tps"] for r in runs])), "decode_tps_8k": r8.get("decode_tps"),
                "prefill_tps_8k": r8.get("prefill_tps"), "must_keep_gib": round(r8["peak"]["peak_footprint_gib"], 2),
                "run_wall_s": [round(r["wall_s"], 1) for r in runs] + [round(r8["wall_s"], 1)], "pageins": r8["peak"]["pageins"]}
    except Exception as exc:
        return {"ok": False, "error": str(exc)[-300:]}


CANDIDATES = [
    ("EXL3 2.08 / MLXL3", lambda: mlxl3(M / "Qwen3.6-35B-A3B-exl3-2.08bpw")),
    ("IQ2_XS GGUF / llama.cpp b12608", lambda: llama(M / "gguf/Qwen_Qwen3.6-35B-A3B-GGUF/Qwen_Qwen3.6-35B-A3B-IQ2_XS.gguf", "llama-b12608")),
    ("Inkling UD-Q2_K_XL / llama.cpp Unsloth", lambda: llama(M / "gguf/Inkling-2-Mini-GGUF/Inkling-2-Mini-UD-Q2_K_XL.gguf", "unsloth-b12588-mix-c0c3de8")),
]
results = []
for target in [float(x) for x in sys.argv[1:]]:
    before = vm()
    size = before["free"] + before["file"] - target
    if size < 1:
        print(f"skip {target}: only {before['free'] + before['file']:.1f} GiB available now"); continue
    b = subprocess.Popen([sys.executable, str(HERE / "ballast.py"), f"{size:.2f}", "1500"], stdout=subprocess.PIPE, text=True)
    try:
        line = b.stdout.readline().strip()
        if not line.startswith("READY"):
            print("ballast failed:", line); continue
        time.sleep(3)
        level = {"available_target_gib": target, "ballast_gib": round(size, 1), "available_measured_gib": round(avail(), 1), "rows": []}
        print(json.dumps({k: v for k, v in level.items() if k != "rows"}), flush=True)
        for name, fn in CANDIDATES:
            s0 = vm()["swap_used_gib"]
            r = fn(); r["candidate"] = name; r["swap_growth_gib"] = round(vm()["swap_used_gib"] - s0, 2)
            r["available_after_gib"] = round(avail(), 1)
            print(json.dumps(r), flush=True); level["rows"].append(r)
        results.append(level)
    finally:
        b.kill(); b.wait()
        time.sleep(5)
(E.RAW_DIR / "pressure_test.json").write_text(json.dumps(results, indent=1))
