"""Shared pieces of the eval: the WikiText-2 slice, prompts, memory probe, metrics.

Every runtime is scored on the same thing:

* PPL / KLD: WikiText-2 test, llama.cpp `llama-perplexity` convention so MLX
  runtimes and GGUF runs are directly comparable. The whole file is tokenized
  once, cut into N_CHUNKS windows of N_CTX tokens, position 0 of each window is
  overwritten with BOS when the tokenizer adds one, and only the second half
  of each window is scored (positions N_CTX/2 .. N_CTX-2 predicting the next
  token). Scored tokens = N_CHUNKS * (N_CTX/2 - 1).
* Speed: first 512 tokens of the same token stream as prompt, 128 new tokens.
* Memory: first 8064 tokens as prompt + 128 new tokens = 8192 context.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import math
import os
import re
import statistics
import time
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
ROOT = EVAL_DIR.parent
WIKI = EVAL_DIR / "wiki.test.raw"
RAW_DIR = ROOT / "results" / "raw"
REF_DIR = EVAL_DIR / "refs"

N_CTX = 2048
N_CHUNKS = int(os.environ.get("EVAL_CHUNKS", "16"))  # override only for smoke tests
SPEED_PROMPT = 512
SPEED_GEN = 128
SPEED_RUNS = 3
MEM_CTX = 8192
LOAD_PROMPT = "The capital of France is"


# ---------------------------------------------------------------- tokens
def wiki_tokens(tokenizer) -> tuple[list[int], int | None]:
    """Token stream for the whole test file, and the BOS id if the model uses one."""
    text = WIKI.read_text(encoding="utf-8")
    hf = getattr(tokenizer, "_tokenizer", tokenizer)
    ids = list(hf.encode(text, add_special_tokens=False))
    bos = getattr(hf, "bos_token_id", None)
    # tokenizer_config.json's add_bos_token is what GGUF conversion follows, so it
    # wins over the tokenizer.json post-processor (MiniCPM5 disagrees with itself:
    # config says no BOS, post-processor adds one). Probe only when it is unset.
    flag = None
    cfg = Path(str(getattr(hf, "name_or_path", ""))) / "tokenizer_config.json"
    if cfg.is_file():
        flag = json.loads(cfg.read_text()).get("add_bos_token")
    if flag is None:
        probe = list(hf.encode("a", add_special_tokens=True))
        flag = len(probe) > 0 and probe[0] == bos
    adds_bos = bool(flag) and bos is not None
    if adds_bos:
        ids = [bos] + ids
    return ids, (bos if adds_bos else None)


def windows(ids: list[int], bos: int | None, n_ctx=N_CTX, n_chunks=N_CHUNKS) -> list[list[int]]:
    if len(ids) < n_ctx * n_chunks:
        raise ValueError(f"test set has {len(ids)} tokens, need {n_ctx * n_chunks}")
    out = []
    for i in range(n_chunks):
        w = list(ids[i * n_ctx : (i + 1) * n_ctx])
        if bos is not None:
            w[0] = bos
        out.append(w)
    return out


def scored_range(n_ctx=N_CTX) -> tuple[int, int]:
    """Logit positions [first, last) that are scored; position j predicts token j+1."""
    return n_ctx // 2, n_ctx - 1


def tokens_sha(wins: list[list[int]]) -> str:
    h = hashlib.sha256()
    for w in wins:
        h.update((",".join(map(str, w)) + "\n").encode())
    return h.hexdigest()[:16]


# ---------------------------------------------------------------- metrics
def summarize_nll(nll_sum: float, n: int) -> dict:
    return {"ppl": math.exp(nll_sum / n), "scored_tokens": n}


def summarize_kld(klds: list[float]) -> dict:
    q = statistics.quantiles(klds, n=100)
    return {
        "kld_mean": sum(klds) / len(klds),
        "kld_median": statistics.median(klds),
        "kld_p95": q[94],
        "kld_p99": q[98],
        "kld_tokens": len(klds),
    }


def median(xs):
    return statistics.median(xs)


# ---------------------------------------------------------------- memory
class _RUsageV2(ctypes.Structure):
    _fields_ = [
        ("ri_uuid", ctypes.c_uint8 * 16),
        ("ri_user_time", ctypes.c_uint64),
        ("ri_system_time", ctypes.c_uint64),
        ("ri_pkg_idle_wkups", ctypes.c_uint64),
        ("ri_interrupt_wkups", ctypes.c_uint64),
        ("ri_pageins", ctypes.c_uint64),
        ("ri_wired_size", ctypes.c_uint64),
        ("ri_resident_size", ctypes.c_uint64),
        ("ri_phys_footprint", ctypes.c_uint64),
        ("ri_proc_start_abstime", ctypes.c_uint64),
        ("ri_proc_exit_abstime", ctypes.c_uint64),
        ("ri_child_user_time", ctypes.c_uint64),
        ("ri_child_system_time", ctypes.c_uint64),
        ("ri_child_pkg_idle_wkups", ctypes.c_uint64),
        ("ri_child_interrupt_wkups", ctypes.c_uint64),
        ("ri_child_pageins", ctypes.c_uint64),
        ("ri_child_elapsed_abstime", ctypes.c_uint64),
        ("ri_diskio_bytesread", ctypes.c_uint64),
        ("ri_diskio_byteswritten", ctypes.c_uint64),
    ]


class _RUsageV4(ctypes.Structure):
    _fields_ = _RUsageV2._fields_ + [
        ("ri_cpu_time_qos_default", ctypes.c_uint64),
        ("ri_cpu_time_qos_maintenance", ctypes.c_uint64),
        ("ri_cpu_time_qos_background", ctypes.c_uint64),
        ("ri_cpu_time_qos_utility", ctypes.c_uint64),
        ("ri_cpu_time_qos_legacy", ctypes.c_uint64),
        ("ri_cpu_time_qos_user_initiated", ctypes.c_uint64),
        ("ri_cpu_time_qos_user_interactive", ctypes.c_uint64),
        ("ri_billed_system_time", ctypes.c_uint64),
        ("ri_serviced_system_time", ctypes.c_uint64),
        ("ri_logical_writes", ctypes.c_uint64),
        ("ri_lifetime_max_phys_footprint", ctypes.c_uint64),
        ("ri_instructions", ctypes.c_uint64),
        ("ri_cycles", ctypes.c_uint64),
        ("ri_billed_energy", ctypes.c_uint64),
        ("ri_serviced_energy", ctypes.c_uint64),
        ("ri_interval_max_phys_footprint", ctypes.c_uint64),
        ("ri_runnable_time", ctypes.c_uint64),
    ]


_libproc = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
GIB = 1024**3


def footprint(pid: int | None = None) -> dict:
    """Current and lifetime-peak physical footprint of a process, in GiB.

    phys_footprint is what Activity Monitor calls Memory; on Apple Silicon it
    includes Metal/GPU allocations, which is where the model weights live.
    File-backed mmap pages are NOT included (see the llama.cpp --no-mmap note).
    """
    pid = os.getpid() if pid is None else pid
    ru = _RUsageV4()
    rc = _libproc.proc_pid_rusage(pid, 4, ctypes.byref(ru))
    if rc != 0:
        raise OSError(ctypes.get_errno(), f"proc_pid_rusage({pid}) failed")
    return {
        "footprint_gib": ru.ri_phys_footprint / GIB,
        "peak_footprint_gib": ru.ri_lifetime_max_phys_footprint / GIB,
        "rss_gib": ru.ri_resident_size / GIB,
        "pageins": ru.ri_pageins,
    }


# ---------------------------------------------------------------- results
_SPLIT = re.compile(r"-\d{5}-of-\d{5}\.gguf$")


def _gguf_parts(p: Path) -> list[Path]:
    """All shards of a split GGUF given any one of them (or just the file itself)."""
    if _SPLIT.search(p.name):
        stem = _SPLIT.sub("", p.name)
        return sorted(f for f in p.parent.glob(stem + "-*-of-*.gguf"))
    return [p]


def dir_size_bytes(path: str | Path) -> int:
    p = Path(path)
    if p.is_file():
        return sum(f.stat().st_size for f in _gguf_parts(p))
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file() and ".cache" not in f.parts)


def weights_bytes(path: str | Path) -> int:
    p = Path(path)
    if p.is_file():
        return sum(f.stat().st_size for f in _gguf_parts(p))
    return sum(
        f.stat().st_size
        for f in p.rglob("*")
        if f.is_file() and f.suffix in (".safetensors", ".gguf") and ".cache" not in f.parts
    )


def result_path(model: str | Path, runtime: str) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    return RAW_DIR / f"{Path(model).name}__{runtime}.json"


def save_section(model: str | Path, runtime: str, section: str, data: dict) -> None:
    """Merge one test's output into the model's raw result JSON."""
    path = result_path(model, runtime)
    doc = json.loads(path.read_text()) if path.exists() else {}
    doc.setdefault("model", str(model))
    doc.setdefault("runtime", runtime)
    doc["size_on_disk_gib"] = dir_size_bytes(model) / GIB
    doc["weights_gib"] = weights_bytes(model) / GIB
    data = dict(data)
    data["at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    doc[section] = data
    path.write_text(json.dumps(doc, indent=2))
