# Stage 2: Qwen3.5-122B-A10B on the M5 Max (128 GB)

EXL3: `turboderp/Qwen3.5-122B-A10B-exl3`, branch `4.00bpw` (e2bf8c62). This is the model itself, so the AREX finetune was not needed.
Baseline: `unsloth/Qwen3.5-122B-A10B-MTP-GGUF`, `UD-Q4_K_M` (3 shards). Same eval settings as Stage 1.
KLD skipped: the bf16 source is 233 GiB (decision recorded with the user).

## Results

| Model | Runtime | Format | bpw (nominal) | bpw (effective) | Size on disk (GiB) | PPL | KLD mean | KLD p99 | Decode tok/s | Prefill tok/s | TTFT (s) | Peak mem @8k (GiB) | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| turboderp EXL3 4.00bpw | pony | EXL3 | 4.00 | 4.07 | 59.41 | 4.622 | n/a | n/a | 27.6 | 427 | 1.267 | 115.43 | peak is the load transient; 77.7 GiB while running at 8k |
| turboderp EXL3 4.00bpw | mlxl3 | EXL3 | 4.00 | 4.07 | 59.41 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | LOAD FAILED: RuntimeError: mlxl3 run exited 2: Error: missing field `partial_rotary_factor` at line 101 column 5 |
| unsloth UD-Q4_K_M | llama | GGUF | n/a | 5.01 | 72.89 | 4.548 | n/a | n/a | 51.4 | 1084 | 0.473 | 73.11 |  |

Effective bpw uses 125.1 B parameters (bf16 repo size / 2 bytes, vision tower included).

## Does it load?

- **PonyExl3: yes.** Loads in 38 s and completes "The capital of France is" with "Paris." Not on its validated list, works anyway.
- **MLXL3: no.** `Error: missing field partial_rotary_factor at line 101 column 5`. The checkpoint's `config.json` keeps that
  value only under `rope_parameters` and has no `tie_word_embeddings` in `text_config`; MLXL3 requires both at the top of
  `text_config`. Fails in 0.2 s, before reading weights. Not patched.
- **llama.cpp: yes.** Loads in 10 s.

## What the numbers say

- Quality is a tie on PPL: 4.62 (EXL3 4.00) vs 4.55 (UD-Q4_K_M, +/- 0.08), with the GGUF 13.6 GiB larger. Top-1 accuracy 63.6% vs 63.8%.
- llama.cpp is about 1.9x faster on decode (51 vs 27 tok/s), 2.5x on prefill at 512 tokens and 7x on prefill at 8k
  (1,290 vs 183 tok/s). PonyExl3's prefill falls as context grows on this model; llama.cpp's does not.
- Memory while running at 8k context: about 78 GiB for PonyExl3 (58.5 GiB resident weights) vs 73 GiB for llama.cpp.

## Pitfalls hit

- **PonyExl3's load transient is about 2x the weights.** Footprint peaks at 115 GiB for a 59 GiB model and the machine went
  24 GB into swap. 54.5 GiB of that is load-time scratch left in MLX's reuse cache; one `mx.clear_cache()` after load
  drops the footprint to 58.9 GiB. Speed is the same either way (27.5 tok/s), so the numbers above are not distorted.
  `EXL3_FUSE_MIN_MB=999999` does not reduce the transient. The README's "load peak near resident" claim does not hold at this size.
- The "optimized" 4.06bpw branch was not downloaded: I assumed it would crash PonyExl3 the way the 2.08bpw Qwen3.6 quant did.
  Correction: that crash comes from the shared expert having a different bit width than the routed experts, so 4.06 may or may not load; untested.
- The GGUF is split in three; pass the `-00001-of-00003.gguf` shard to llama.cpp.
