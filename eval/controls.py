#!/usr/bin/env python3
"""Negative controls: show the KLD check can fail, and passes when it should.

  controls.py <reference_name> <bf16_model_dir> <other_reference_name>

1. Self:      the bf16 model re-scored against its own saved reference must give KLD ~ 0
              (exercises the save/load/fp16 path; anything above 1e-3 means the pipeline is lossy).
2. Wrong ref: scoring against a reference built on a different token slice must be refused,
              not silently scored.
3. Shift:     the reference compared with itself shifted by one position must give a large
              KLD (an off-by-one alignment bug would look like this, so it must not look small).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

import evallib as E

HERE = Path(__file__).resolve().parent


def main() -> int:
    ref, model, other = sys.argv[1], str(Path(sys.argv[2]).resolve()), sys.argv[3]
    out, ok = {}, True

    p = subprocess.run([sys.executable, str(HERE / "worker_mlx.py"), "ppl", model, "mlx", "--ref", ref],
                       cwd=HERE, capture_output=True, text=True)
    kld = next((json.loads(l) for l in p.stdout.splitlines() if l.startswith('{"kld_mean"')), None)
    out["self_kld_mean"] = kld and kld["kld_mean"]
    out["self_pass"] = bool(kld) and abs(kld["kld_mean"]) < 1e-3
    ok &= out["self_pass"]

    p = subprocess.run([sys.executable, str(HERE / "worker_mlx.py"), "ppl", model, "mlx", "--ref", other],
                       cwd=HERE, capture_output=True, text=True)
    out["wrong_ref_exit"] = p.returncode
    out["wrong_ref_pass"] = p.returncode != 0 and "different token slice" in (p.stderr + p.stdout)
    ok &= out["wrong_ref_pass"]

    a = np.load(E.REF_DIR / ref / "00.npy").astype(np.float32)
    shifted = float((np.exp(a[:-1]) * (a[:-1] - a[1:])).sum(-1).mean())
    out["shift_by_one_kld_mean"] = shifted
    out["shift_pass"] = shifted > 1.0
    ok &= out["shift_pass"]

    out["all_pass"] = bool(ok)
    print(json.dumps(out, indent=2))
    (E.RAW_DIR / f"controls__{ref}.json").write_text(json.dumps(out, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
