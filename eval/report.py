#!/usr/bin/env python3
"""Turn raw eval JSON into the stage table.

  report.py <stage_manifest.json>            # prints markdown to stdout

Manifest: {"params": <parameter count of the source model>,
           "rows": [{"model": "<path>", "runtime": "pony", "label": "...", "format": "EXL3", "bpw": 4.0}, ...]}

`bpw` is the nominal figure (what the quant is called). The table also shows the
effective bpw = weight bytes * 8 / params, which is what actually has to fit in
memory: EXL3's nominal bpw leaves out embeddings and the output head.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import evallib as E


def f(v, fmt="{:.2f}", na="n/a"):
    return na if v is None else fmt.format(v)


def main() -> None:
    man = json.loads(Path(sys.argv[1]).read_text())
    params = man.get("params")
    head = ("| Model | Runtime | Format | bpw (nominal) | bpw (effective) | Size on disk (GiB) | PPL | KLD mean | KLD p99 "
            "| Decode tok/s | Prefill tok/s | TTFT (s) | Peak mem @8k (GiB) | Notes |")
    print(head)
    print("|" + "---|" * (head.count("|") - 1))
    for row in man["rows"]:
        p = E.result_path(Path(row["model"]).resolve(), row["runtime"])
        d = json.loads(p.read_text()) if p.exists() else {}
        load, ppl, kld, spd, mem = (d.get(k, {}) for k in ("load", "ppl", "kld", "speed", "mem"))
        notes = []
        if not d:
            notes.append("not run")
        if load.get("ok") is False:
            notes.append("LOAD FAILED: " + load.get("error", "").strip().splitlines()[-1][:160])
        elif load and not load.get("sane", True):
            notes.append("loads, completion not sane")
        for k in ("ppl", "speed", "mem"):
            if d.get(k, {}).get("ok") is False:
                notes.append(f"{k} failed")
        if d.get("budget", {}).get("over"):
            notes.append(f"over {d['budget']['budget_gib']} GiB budget: not deployable")
        if row.get("note"):
            notes.append(row["note"])
        size = E.dir_size_bytes(row["model"]) / E.GIB if Path(row["model"]).exists() else None
        wbytes = E.weights_bytes(row["model"]) if Path(row["model"]).exists() else None
        rp = row.get("params", params)  # per-row override for a different base model
        eff = wbytes * 8 / rp if (wbytes and rp) else None
        print("| " + " | ".join([
            row["label"], row["runtime"], row["format"], f(row.get("bpw")), f(eff), f(size),
            f(ppl.get("ppl"), "{:.3f}"), f(kld.get("kld_mean"), "{:.4f}"), f(kld.get("kld_p99"), "{:.3f}"),
            f(spd.get("decode_tps"), "{:.1f}"), f(spd.get("prefill_tps"), "{:.0f}"), f(spd.get("ttft_s"), "{:.3f}"),
            f(mem.get("peak_footprint_gib")), "; ".join(notes),
        ]) + " |")


if __name__ == "__main__":
    main()
