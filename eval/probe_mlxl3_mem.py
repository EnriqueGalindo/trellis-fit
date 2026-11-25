"""Footprint of `mlxl3 run` over time for a short prompt. Usage: probe_mlxl3_mem.py <model_dir> [max_tokens]"""
import json, subprocess, sys, time
from pathlib import Path
import evallib as E
from worker_mlxl3 import base
model = str(Path(sys.argv[1]).resolve()); n = sys.argv[2] if len(sys.argv) > 2 else "400"
p = subprocess.Popen(base("run", model, "--prompt", "Write a long, detailed essay about the history of bridges.", "--max-tokens", n),
                     stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.DEVNULL, text=True)
samples = []
while p.poll() is None:
    try: samples.append(E.footprint(p.pid))
    except OSError: pass
    time.sleep(0.05)
out, err = p.communicate()
fp = sorted(s["footprint_gib"] for s in samples)
print(json.dumps({"exit": p.returncode, "err": err.strip()[-300:], "samples": len(fp), "peak_gib": round(max(s["peak_footprint_gib"] for s in samples), 3),
                  "median_gib_while_generating": round(fp[len(fp) // 2], 3), "p90_gib": round(fp[int(len(fp) * .9)], 3), "stats": out.strip().splitlines()[-1][:120]}))
