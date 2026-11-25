#!/usr/bin/env python3
"""llama.cpp with MoE experts kept host-side (mmap, evictable) and only a cache of them on the GPU.

  probe_moe_cache.py <model.gguf> <cache_mib|all-gpu|cpu-only> ...

Reports speed (512-token prompt, 128 generated, median of 3) and memory through an 8k context:
  footprint = memory the process must keep (GPU buffers, KV cache); file-backed weights excluded
  rss       = everything resident right now, including mmap'd weight pages that have been touched
On this 128 GB machine nothing is ever evicted, so speed here is the best case: no SSD re-reads.
"""
import json, os, sys
from pathlib import Path
import evallib as E
import worker_llama as W

gguf = str(Path(sys.argv[1]).resolve())
rows = []
for mode in sys.argv[2:]:
    extra = {"all-gpu": "", "cpu-only": "--cpu-moe"}.get(mode, f"--cpu-moe --moe-cache-mib {mode}")
    os.environ["LLAMA_EXTRA_ARGS"] = extra
    srv = W.Server(gguf, E.MEM_CTX)  # default load mode: mmap
    try:
        srv.wait()
        p512 = W.wiki_prefix_tokens(srv, E.SPEED_PROMPT)
        srv.generate(p512, 16)
        runs = [srv.generate(p512, E.SPEED_GEN) for _ in range(E.SPEED_RUNS)]
        after_short = E.footprint(srv.proc.pid)
        r8 = srv.generate(W.wiki_prefix_tokens(srv, E.MEM_CTX - E.SPEED_GEN), E.SPEED_GEN)
        end = E.footprint(srv.proc.pid)
        row = {"mode": mode, "args": extra or "(default)",
               "decode_tps": round(E.median([r["decode_tps"] for r in runs]), 1),
               "prefill_tps": round(E.median([r["prefill_tps"] for r in runs]), 0),
               "decode_tps_8k": round(r8["decode_tps"], 1), "prefill_tps_8k": round(r8["prefill_tps"], 0),
               "footprint_gib_after_512": round(after_short["footprint_gib"], 2), "rss_gib_after_512": round(after_short["rss_gib"], 2),
               "peak_footprint_gib_8k": round(end["peak_footprint_gib"], 2), "rss_gib_8k": round(end["rss_gib"], 2)}
    except Exception as exc:
        row = {"mode": mode, "args": extra, "error": str(exc)[-400:]}
    finally:
        srv.stop()
    print(json.dumps(row), flush=True)
    rows.append(row)
(E.RAW_DIR / f"moe_cache__{Path(gguf).name}.json").write_text(json.dumps(rows, indent=1))
