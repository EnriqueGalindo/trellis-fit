#!/usr/bin/env python3
"""Run the full eval on one model in one runtime.

  run_eval.py <model_path> <runtime> [--ref NAME] [--save-ref NAME]
              [--tests load,ppl,speed,mem] [--budget-gib 10.5]

  runtime   pony   EXL3 directory through PonyExl3
            mlxl3  EXL3 directory through the MLXL3 engine
            llama  .gguf file through llama.cpp
            mlx    bf16 source directory through mlx-lm (use --save-ref to make it the KLD reference)

  --ref NAME       compute KLD against eval/refs/NAME (made earlier with --save-ref)
  --budget-gib X   flag the model as not deployable if its weights exceed X GiB

Each test runs in its own process. If the load check fails the remaining tests are
skipped and the error is recorded. Results merge into results/raw/<model>__<runtime>.json.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import evallib as E

HERE = Path(__file__).resolve().parent
WORKER = {"pony": "worker_mlx.py", "mlx": "worker_mlx.py", "mlxl3": "worker_mlxl3.py", "llama": "worker_llama.py"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("runtime", choices=sorted(WORKER))
    ap.add_argument("--ref")
    ap.add_argument("--save-ref")
    ap.add_argument("--tests", default="load,ppl,speed,mem")
    ap.add_argument("--budget-gib", type=float)
    ap.add_argument("--timeout", type=int, default=6 * 3600, help="seconds per test")
    a = ap.parse_args()

    model = str(Path(a.model).resolve())
    if not Path(model).exists():
        raise SystemExit(f"no such model path: {model}")
    w_gib = E.weights_bytes(model) / E.GIB
    print(f"== {Path(model).name} [{a.runtime}] weights {w_gib:.2f} GiB")
    if a.budget_gib is not None:
        over = w_gib > a.budget_gib
        E.save_section(model, a.runtime, "budget", {"budget_gib": a.budget_gib, "weights_gib": w_gib, "over": over})
        if over:
            print(f"!! NOT DEPLOYABLE: weights {w_gib:.2f} GiB exceed the {a.budget_gib} GiB budget (evaluating anyway)")

    failed = False
    for test in [t.strip() for t in a.tests.split(",") if t.strip()]:
        cmd = [sys.executable, str(HERE / WORKER[a.runtime]), test, model]
        if a.runtime in ("pony", "mlx"):
            cmd.append(a.runtime)
        if test == "ppl" and a.ref:
            cmd += ["--ref", a.ref]
        if test == "ppl" and a.save_ref and a.runtime == "mlx":
            cmd += ["--save-ref", a.save_ref]
        print(f"-- {test}", flush=True)
        t0 = time.perf_counter()
        try:
            p = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True, timeout=a.timeout)
            rc, out, err = p.returncode, p.stdout, p.stderr
        except subprocess.TimeoutExpired as exc:
            rc, out, err = -1, "", f"timed out after {a.timeout}s\n{exc.stderr or ''}"
        dt = time.perf_counter() - t0
        if rc != 0:
            failed = True
            tail = "\n".join((err or out).strip().splitlines()[-25:])
            print(f"   FAILED (exit {rc}, {dt:.0f}s)\n{tail}")
            E.save_section(model, a.runtime, test, {"ok": False, "exit": rc, "error": tail, "seconds": dt})
            if test == "load":
                print("   load check failed: skipping the remaining tests")
                break
            continue
        for line in out.strip().splitlines():
            if line.startswith("{"):
                d = json.loads(line)
                print("   " + ", ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                                        for k, v in d.items() if k not in ("runs", "prompt")))
    print(f"-> {E.result_path(model, a.runtime)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
