# Stage 1: pipeline shakedown on MiniCPM5-1B

Machine: M5 Max, 128 GB, macOS 26.7. Source: `openbmb/MiniCPM5-1B` (revision ac83f81), 1.08 B parameters.
Tools: PonyExl3 8e7fa6b (v0.3.0, mlx 0.32.4), MLXL3 engine 2.0.19, llama.cpp b12359.

## Eval settings (identical for every row)

- PPL/KLD: WikiText-2 test, 16 windows of 2048 tokens, second half of each window scored (16,368 tokens), llama.cpp convention.
- KLD: KL(bf16 || quant) per token against the bf16 source run in mlx-lm, including for GGUF.
- Speed: 512-token prompt, 128 generated, greedy, median of 3 after one warmup.
- Peak memory: process physical footprint through an 8,192-token context (GGUF loaded without mmap).

## Results

| Model | Runtime | Format | bpw (nominal) | bpw (effective) | Size on disk (GiB) | PPL | KLD mean | KLD p99 | Decode tok/s | Prefill tok/s | TTFT (s) | Peak mem @8k (GiB) | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MiniCPM5-1B source | mlx | bf16 | 16.00 | 16.00 | 2.02 | 21.659 | n/a | n/a | 252.0 | 25423 | 0.022 | 3.32 | KLD reference |
| ponyexl3-convert 4.0 | pony | EXL3 | 4.00 | 6.61 | 0.84 | 22.640 | 0.0576 | 0.357 | 198.6 | 13175 | 0.045 | 2.21 |  |
| ponyexl3-convert 2.0 | pony | EXL3 | 2.00 | 5.35 | 0.68 | 65.894 | 1.2124 | 5.138 | 206.1 | 13449 | 0.044 | 2.05 | loads, completion not sane |
| turboderp 4.00 (exllamav3) | pony | EXL3 | 4.00 | 6.61 | 0.84 | 22.154 | 0.0316 | 0.194 | 205.9 | 13414 | 0.044 | 2.58 | published quant, for comparison |
| turboderp 2.00 (exllamav3) | pony | EXL3 | 2.00 | 5.35 | 0.68 | 36.461 | 0.5652 | 3.229 | 207.2 | 13175 | 0.045 | 2.43 | published quant, for comparison |
| ponyexl3-convert 4.0 | mlxl3 | EXL3 | 4.00 | 6.61 | 0.84 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | LOAD FAILED: RuntimeError: mlxl3 run exited 2: Error: native chat does not support model type "llama" |
| ponyexl3-convert 2.0 | mlxl3 | EXL3 | 2.00 | 5.35 | 0.68 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | LOAD FAILED: RuntimeError: mlxl3 run exited 2: Error: native chat does not support model type "llama" |
| openbmb Q4_K_M | llama | GGUF | n/a | 5.09 | 0.64 | 23.936 | 0.1050 | 0.614 | 424.7 | 19910 | 0.026 | 1.28 | llama.cpp path shakedown |
| openbmb F16 | llama | GGUF | 16.00 | 16.04 | 2.02 | 21.667 | -0.0003 | 0.002 | 235.5 | 20566 | 0.025 | 2.89 | cross-runtime noise floor |

"bpw (effective)" is weight bytes x 8 / parameters. It is far above nominal on this model because the
embedding table (200 M of 1,080 M parameters) stays bf16 and the output head is quantized at 6 bits.

## What the numbers say

- 4.0 to 2.0 bpw with `ponyexl3-convert`: PPL 22.6 to 65.9, KLD 0.058 to 1.21, top-1 agreement with bf16 86% to 48%.
  The 2.0 bpw quant fails the load check (it no longer answers "Paris"). Disk size drops only 19%.
- `ponyexl3-convert` is clearly worse than turboderp's published exllamav3 quants at the same bitrate:
  KLD 0.058 vs 0.032 at 4.0, and 1.21 vs 0.57 at 2.0. The README's parity claim is for its
  "advanced, oracle-comparable" path, not the one-command path used here.
- Speed and memory barely move between 4.0 and 2.0 bpw on a model this small.
- Cross-runtime noise floor: the F16 GGUF in llama.cpp scores KLD 0.000 and PPL 21.667 against 21.659 in mlx-lm,
  so GGUF and EXL3 quality numbers are comparable. Speeds across runtimes are not like-for-like on a 1 B model.

## Conversion

| Target | Wall time | Peak memory | Output |
|---|---|---|---|
| 4.0 bpw | 4 min 30 s | 13.0 GiB | 0.84 GiB |
| 2.0 bpw | 8 min 02 s (evals were running alongside) | 12.9 GiB | 0.68 GiB |

## What failed and why

- **MLXL3 cannot run MiniCPM5 at all.** `Error: native chat does not support model type "llama"`. MLXL3 has no Llama-layout
  runtime, so Stage 1 has no MLXL3 numbers.
- **MLXL3 also rejects `UnstableLlama/Qwen3.5-2B-exl3-4.00bpw`** (tried as a small substitute):
  `missing field partial_rotary_factor`; with that value supplied it still refuses because the model ties
  input and output embeddings.
- **MLXL3 path tested instead on `UnstableLlama/Qwen3.6-35B-A3B-exl3-2.08bpw`** (the Stage 3 primary): load, speed,
  memory and a 2-window PPL run all work. Full numbers belong to Stage 3.
- **PonyExl3 cannot load that 2.08 bpw quant.** `ValueError: [concatenate] All the input array dimensions must match`
  in `_stack_expert_switch_glu`: PonyExl3 stacks each layer's routed experts together with the shared expert and assumes one bit width;
  this quant has routed experts at 2 bits and the shared expert at 4 (corrected after the Stage 3 byte breakdown). Not patched, per instructions. This affects Stage 3.

## Pipeline problems fixed here

- MiniCPM5's tokenizer files disagree about BOS (config says no, tokenizer.json adds one). The eval now follows
  `tokenizer_config.json`, which is what the GGUF does, so all runtimes score identical tokens.
- llama.cpp b12359 no longer accepts `--no-mmap`; it is `--load-mode none`.
- The llama.cpp "latest" release (v1.0.0) ships no binaries; they are under the nightly tag named in its `nightly-tag.txt`.
- upstream `mlx_eval.compare` cannot load EXL3 checkpoints, so KLD is computed by the scripts in `eval/` with the same formula.
- MLXL3's CLI always applies the chat template and stops at EOS, so its speed prompt carries a few template tokens.
