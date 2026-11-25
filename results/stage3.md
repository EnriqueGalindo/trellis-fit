# Stage 3: the mini-sized model (Qwen3.6-35B-A3B, 10.5 GiB weight limit)

Measured on the M5 Max. Speeds are for relative comparison only; the 16 GB mini is much slower.
Same eval settings as Stages 1 and 2. KLD reference: `Qwen/Qwen3.6-35B-A3B` bf16 in mlx-lm (PPL 5.903).

## Results

| Model | Runtime | Format | bpw (nominal) | bpw (effective) | Size on disk (GiB) | PPL | KLD mean | KLD p99 | Decode tok/s | Prefill tok/s | TTFT (s) | Peak mem @8k (GiB) | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3.6-35B-A3B source | mlx | bf16 | 16.00 | 16.00 | 66.99 | 5.903 | 0.0000 | 0.001 | 71.6 | 2978 | 0.185 | 69.14 | KLD reference |
| Qwen3.6 EXL3 2.08 (primary) | mlxl3 | EXL3 | 2.08 | 2.51 | 10.57 | 6.302 | 0.1141 | 1.199 | 121.7 | 1816 | 0.288 | 11.48 | over 10.5 GiB budget: not deployable; 11,276,374,187 bytes: 2 MB over a strict 10.5 GiB |
| Qwen3.6 EXL3 2.08 (primary) | pony | EXL3 | 2.08 | 2.51 | 10.57 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | LOAD FAILED: ValueError: [concatenate] All the input array dimensions must match exactly except for the concatenation axis. However, the provided shapes are (128,32,32), (12 |
| Qwen3.6 EXL3 2.49 | mlxl3 | EXL3 | 2.49 | 2.89 | 12.17 | 6.212 | 0.0861 | 0.906 | 119.9 | 1797 | 0.290 | 13.08 | over 10.5 GiB budget: not deployable |
| Qwen3.6 EXL3 2.49 | pony | EXL3 | 2.49 | 2.89 | 12.17 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | LOAD FAILED: ValueError: [concatenate] All the input array dimensions must match exactly except for the concatenation axis. However, the provided shapes are (128,32,48), (12; over 10.5 GiB budget: not deployable |
| Qwen3.6 IQ2_XS | llama-b12359 | GGUF | n/a | 2.60 | 10.89 | 6.617 | 0.1671 | 1.577 | 135.8 | 3365 | 0.153 | 11.10 | over 10.5 GiB budget: not deployable; mainline b12359 |
| Qwen3.6 IQ2_XS | llama | GGUF | n/a | 2.60 | 10.89 | 6.617 | 0.1671 | 1.577 | 117.4 | 2857 | 0.180 | 10.75 | over 10.5 GiB budget: not deployable; Unsloth build b12588 |
| Inkling-2-Mini UD-Q2_K_XL | llama | GGUF | n/a | 2.77 | 8.51 | 5.989 | n/a | n/a | 39.8 | 924 | 0.555 | 10.32 | Unsloth build b12588; different model, so no KLD and PPL is not comparable across models |
| Inkling-2-Mini UD-IQ3_XXS | llama | GGUF | n/a | 3.53 | 10.86 | 5.819 | n/a | n/a | 39.2 | 911 | 0.563 | 12.67 | over 10.5 GiB budget: not deployable; Unsloth build b12588; different model, so no KLD and PPL is not comparable across models |

Effective bpw = weight bytes x 8 / parameters (35.95 B for Qwen3.6-35B-A3B incl. vision tower, 26.4 B for Inkling).
The budget flag is a strict comparison against 10.5 GiB of weight files.

## What the numbers say

- **Best quality inside the limit: the 2.08 bpw EXL3 quant.** KLD 0.114 and PPL 6.30 (bf16 5.90), top-1 agreement with bf16 85.9%.
  It beats the IQ2_XS GGUF (KLD 0.167, PPL 6.62, 83.2%) while being 0.39 GiB smaller.
- **2.49 bpw buys little:** KLD 0.086, PPL 6.21, for 1.6 GiB more. The curve from 2.08 to 2.49 is shallow, nothing like the 1 B model's collapse in Stage 1.
- **Speed is close between EXL3 in MLXL3 and GGUF in llama.cpp** on decode (122 vs 117 to 136 tok/s). llama.cpp prefill is 1.6 to 1.9x faster at 512 tokens.
- **Memory at 8k context: 11.5 GiB (EXL3 2.08, MLXL3) vs 10.8 to 11.1 GiB (IQ2_XS).** MLXL3 has no load transient to speak of.
- **Inkling-2-Mini UD-Q2_K_XL is the only candidate clearly under the limit (8.51 GiB)**, with 10.3 GiB peak at 8k. It decodes at 40 tok/s here,
  a third of the Qwen quants, because its middle layers run three times. Its PPL (5.99) is on a different model and tokenizer, so it is
  not a quality ranking against Qwen. Its model card recommends 32 GB Macs.

## Deployability against the 10.5 GiB limit

| Candidate | Weights | Verdict |
|---|---|---|
| Qwen3.6 EXL3 2.08 bpw | 10.502 GiB | at the limit (2 MB over a strict reading) |
| Qwen3.6 EXL3 2.49 bpw | 12.10 GiB | not deployable |
| Qwen3.6 IQ2_XS GGUF | 10.89 GiB | not deployable |
| Inkling UD-Q2_K_XL | 8.51 GiB | fits |
| Inkling UD-IQ3_XXS | 10.86 GiB | not deployable |

## Negative controls (eval/controls.py)

All pass: bf16 scored against its own saved reference gives KLD 0.0000019; a reference built on other tokens is refused;
a one-position misalignment gives KLD 10.7.

## What failed and why

- **PonyExl3 cannot load either Qwen3.6 EXL3 quant.** `ValueError: [concatenate] All the input array dimensions must match` in
  `_stack_expert_switch_glu`. PonyExl3 stacks each layer's routed experts together with the shared expert and assumes one bit width.
  These quants keep the shared expert at 4 bits while routed experts are at 2 (2.08) or 2 and 3 (2.49). Its validated 4.00 bpw quant is 4 bits throughout. So the EXL3 rows are MLXL3 only. Not patched.
- **Inkling does not load in mainline llama.cpp.** b12359: `unknown model architecture: 'inkling2'` (the build predates the model).
  b12608: `check_tensor_dims: tensor 'blk.24.attn_q.weight' not found` (mainline expects unrolled layers). It loads in Unsloth's
  fork build `b12588-mix-c0c3de8`, as the Unsloth guide says. The IQ2_XS baseline was rerun on that build: identical PPL/KLD,
  14% slower decode than mainline b12359.
- **Downloads:** unauthenticated Hugging Face downloads stalled silently on a network change and do not resume partial files;
  the bf16 download also died once with a CDN error. Fixed by logging in and wrapping the download in a retry loop.
- **A background job reported success while the download inside it had failed** (my command chain ended in `echo`). Caught when
  the reference build failed on missing weights.
- One speed sample for 2.08 bpw read 91 tok/s; two repeat runs gave 122. The table has the repeat.

## Not done

- Optional `ponyexl3-convert` of Qwen3.6-35B-A3B at 2.25 bpw: needs approval, see the stage summary.
