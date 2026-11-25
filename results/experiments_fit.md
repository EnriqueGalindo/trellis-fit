# Experiments: can the Qwen3.6-35B-A3B quant be made to fit better?

Run on the M5 Max (128 GB) after Stage 3. Scripts are in `eval/`; raw output in `results/raw/`.

## 1. Byte breakdown of the 2.08 bpw EXL3 quant (`byte_breakdown.py`)

| Component | GiB | Share | Bits per weight |
|---|---|---|---|
| Routed experts (MoE) | 7.647 | 72.9% | 2 |
| Token embeddings | 0.947 | 9.0% | 16 |
| Vision tower | 0.832 | 7.9% | 16 |
| Linear attention | 0.479 | 4.6% | 4 |
| Output head | 0.356 | 3.4% | 6 |
| Full attention | 0.127 | 1.2% | 4 |
| Shared expert | 0.059 | 0.6% | 4 |
| MoE router, norms | 0.039 | 0.4% | 16 |

Every routed expert is at exactly 2 bits. The 2.49 bpw quant differs only in moving some layers' experts to 3 bits.

## 2. Vision tower (`strip_tensors.py`, `probe_mlxl3_mem.py`)

A copy of the quant without the 333 vision tensors (0.832 GiB smaller, 9.67 GiB of weights) loads and runs in MLXL3.

| Variant | Footprint while generating (GiB) | Peak (GiB) | Decode tok/s |
|---|---|---|---|
| Original | 10.30 | 10.41 | 123, 120 |
| Vision tower removed | 10.30 | 10.41 | 123, 121 |

MLXL3 already skips the vision weights. Removing them saves disk only, not memory.

## 3. Is expert routing concentrated? (`expert_usage.py`, bf16 model in mlx-lm, 8,192 tokens per text)

| Measure | WikiText-2 | Python source code |
|---|---|---|
| Experts touched after 128 tokens | 48.7% | 49.2% |
| Experts touched after 512 tokens | 68.2% | 84.0% |
| Experts touched after 8,192 tokens | 96.3% | 99.3% |
| Experts needed for 90% of activations | 42.4% | 47.2% |
| Experts needed for 99% of activations | 73.2% | 80.6% |

Simulated least-recently-used cache over (layer, expert) pairs, tokens in decode order:

| Cache holds | Expert loads per token (WikiText) | (code) | Memory saved on the 2.08 quant |
|---|---|---|---|
| 75% of experts | 1.8 | 5.1 | about 1.9 GiB |
| 50% | 13.3 | 20.2 | about 3.8 GiB |
| 25% | 57 | 62 | about 5.7 GiB |
| 10% | 123 | 132 | about 6.9 GiB |

One expert is about 0.75 MB at 2 bits. Routing is spread out: nearly every expert is used within a few thousand tokens,
but a quarter of them are rare enough that leaving them on disk costs a few small reads per token.

## 4. llama.cpp's expert cache (`probe_moe_cache.py`, IQ2_XS GGUF, build b12608, mmap load)

| Mode | Decode tok/s | Prefill tok/s | Must-keep memory at 8k (GiB) | Resident incl. mapped weights (GiB) |
|---|---|---|---|---|
| Default (all on GPU) | 127 | 3,298 | 0.84 | 10.65 |
| Experts host-side, 6 GiB cache | 49 | 138 | 6.89 | 16.83 |
| Experts host-side, 4 GiB cache | 46 | 155 | 4.89 | 14.83 |
| Experts host-side, 2 GiB cache | 38 | 243 | 2.90 | 12.83 |
| Experts host-side, 1 GiB cache | 33 | 388 | 1.90 | 11.83 |
| Experts host-side, no cache | 26 | 697 | 0.90 | 10.82 |

- The explicit cache is counterproductive on Apple Silicon: it copies experts into a second buffer (more memory, not less)
  and decode drops 2.6x to 5x. It is built for machines with a separate GPU.
- The default mode is the interesting one: with mmap, llama.cpp's weights are file-backed. Only 0.84 GiB is memory the
  process must keep; the other 10 GiB can be dropped and re-read by macOS under pressure. MLXL3 holds its 10.3 GiB as
  memory it must keep.
- Not measured: what happens when macOS actually has to evict. This machine never does. Section 3 predicts a cost of
  a few expert re-reads per token for a shortfall up to about 2 GiB, growing quickly beyond that.

## What this changes

- The earlier memory comparison (GGUF 11.1 GiB vs EXL3 11.5 GiB) loaded the GGUF without mmap. That is the right number
  for "how much it wants resident" and the wrong one for "what happens on a 16 GB machine": the GGUF can shed pages, the
  EXL3 runtime cannot.
- For the EXL3 route, the only block with real room is the routed experts (73% of bytes, already at 2 bits).
- Both remaining questions need the mini, or deliberate memory pressure on this machine.

## 5. Restricting the router to the most-used experts (`expert_prune.py`, bf16 model in mlx-lm)

Usage counted on 16k tokens of WikiText-2 train plus 16k tokens of Python source; scored on the standard WikiText-2
test windows against the unmodified bf16 reference, and on held-out Python source (PPL only).
Dropped experts are masked out of the router, so the 8 experts per token are chosen among the kept ones.
"Per layer" keeps the same share in every layer.

| Experts kept (per layer) | Share of calibration activations they covered | PPL | KLD vs bf16 | Top-1 agreement | Held-out code PPL | Expert bytes freed on the 2.08 quant |
|---|---|---|---|---|---|---|
| 100% | 100% | 5.903 | 0.000 | 100% | 2.587 | 0 |
| 90% | 99.0% | 5.932 | 0.023 | 94.8% | 2.581 | 0.76 GiB |
| 75% | 95.4% | 6.226 | 0.089 | 90.5% | 2.627 | 1.91 GiB |
| 60% | 89.4% | 6.752 | 0.175 | 86.2% | 2.699 | 3.06 GiB |
| 50% | 83.7% | 7.438 | 0.276 | 82.1% | 2.729 | 3.82 GiB |
| 25% | 60.5% | 11.861 | 0.797 | 66.1% | 3.117 | 5.74 GiB |

Ranking all (layer, expert) pairs together instead of per layer gives nearly the same numbers (KLD 0.020, 0.088, 0.158, 0.264, 0.796).

For scale: quantizing to 2.49 bpw costs KLD 0.086 and to 2.08 bpw costs 0.114.

- Dropping the least-used 10% of experts is nearly free (KLD 0.023). Dropping 25% costs about as much as quantizing to 2.49 bpw.
- "Under half the experts do 90% of the work" does not mean the other half can go: keeping 60% (89% of activations) costs KLD 0.175,
  worse than the whole 2.08 bpw quantization. The rarely used experts matter when they are used.
- Not tested: the same masking on a quantized model. The two losses probably add, but that is an assumption.
- No runtime we have can load a checkpoint with experts missing, so the bytes are only freed with a runtime change, or for free
  under mmap, where never-used experts are never read from disk.

## 6. Imitating a 16 GB machine (`ballast.py`, `pressure_test.py`)

A helper process pinned memory so only the stated amount was left as free or file cache. 8k-context speed is the figure that
shows pressure; 512-token speeds were unchanged except where noted.

| Left available | EXL3 2.08 in MLXL3 (needs 11.5 GiB it cannot give back) | IQ2_XS GGUF in llama.cpp (0.84 GiB it cannot give back) | Inkling UD-Q2_K_XL in llama.cpp (2.0 GiB) |
|---|---|---|---|
| no limit | 106 tok/s decode at 8k, 3,261 prefill | 118 to 127 | 37 to 40 |
| 12.5 GiB | 106 decode, 3,031 prefill | 125 decode, 3,802 prefill | 38 decode |
| 10.5 GiB | 106 decode, 2,529 prefill, 0.7 GiB pushed to swap | 124 decode, 3,773 prefill | 38 decode |
| 8.5 GiB | **37 decode, 1,006 prefill**, 512-token prefill fell from 1,664 to 280, 3.2 GiB pushed to swap | 122 decode, 3,697 prefill | 37 decode |
| 6.5 GiB | not reached: macOS refused to pin more than 102 GiB | | |

- The GGUF in llama.cpp did not slow down at any level reached. The EXL3 quant in MLXL3 held up while other apps could be pushed
  to swap to make room, and lost two thirds of its speed once they could not.
- The imitation is softer than a real mini. This laptop had about 20 GiB of other apps' memory that macOS compressed or swapped
  to make room (available memory rose 3 GiB after each test started). It also keeps this machine's GPU memory cap, chip and SSD.
- So: this shows the direction (file-mapped weights degrade gently, held weights do not), not the mini's actual numbers.

## 7. Our own quant: least-used experts disabled in real model files (`bake_router.py`)

Only the small router matrices are edited: each dropped expert's row is replaced so its score is hugely negative for
every token. The copies run in the unmodified runtimes. Dropped experts were selected 0% of the time.

| Variant | Runtime | PPL | KLD vs bf16 | Change in KLD | Decode tok/s |
|---|---|---|---|---|---|
| EXL3 2.08, unedited | MLXL3 | 6.302 | 0.114 | | 122 |
| EXL3 2.08, least-used 10% disabled | MLXL3 | 6.354 | 0.126 | +0.012 | 121 |
| EXL3 2.08, least-used 25% disabled | MLXL3 | 6.695 | 0.182 | +0.067 | 121 |
| IQ2_XS GGUF, unedited | llama.cpp b12608 | 6.617 | 0.167 | | 133 |
| IQ2_XS GGUF, least-used 10% disabled | llama.cpp b12608 | 6.679 | 0.179 | +0.012 | 133 |
| IQ2_XS GGUF, least-used 25% disabled | llama.cpp b12608 | 7.012 | 0.233 | +0.066 | 132 |

- On a quantized model the cost of dropping experts is a little smaller than on bf16 (+0.012 vs +0.023 for 10%), so the
  two losses do not simply add.
- The files are the same size and use the same memory. Under file mapping the edited GGUFs show the same 10.8 GiB resident
  as the original: llama.cpp reads the whole file at load. **Correction to section 5:** never-used experts are not skipped
  for free. Freeing the bytes needs a file with those experts removed and a runtime that accepts fewer experts per layer.

## 8. Embeddings at fewer bits (`sim_levers.py`, simulated on the bf16 model)

| Embedding precision | PPL | KLD vs bf16 | Saving on the EXL3 quant |
|---|---|---|---|
| 16 bits (as shipped) | 5.903 | 0.000 | |
| 8 bits | 5.898 | 0.005 | 0.47 GiB |
| 6 bits | 5.892 | 0.007 | 0.59 GiB |
| 4 bits | 6.019 | 0.035 | 0.71 GiB |

The GGUF already does this (its embedding table is 0.155 GiB against 0.947 GiB in the EXL3 quant). The EXL3 format and
both EXL3 runtimes store embeddings at 16 bits, so using this needs a format or runtime change.

## 9. Experts below 2 bits (`sim_levers.py`, simulated with scalar stand-in quantizers)

EXL3 allows 1 to 8 bits per weight in both runtimes. The stand-ins have no calibration and are worse than trellis coding:
per the QTIP paper a 1-bit trellis has error 0.280 against 0.363 for the 1-bit stand-in, and 0.073 against 0.118 at 2 bits.

| Configuration | PPL | KLD vs bf16 | Expert bytes saved vs all-2-bit |
|---|---|---|---|
| Least-used 10% at 1 bit, rest untouched | 5.910 | 0.015 | 0.38 GiB |
| Least-used 25% at 1 bit, rest untouched | 6.028 | 0.053 | 0.96 GiB |
| Least-used 50% at 1 bit, rest untouched | 6.660 | 0.165 | 1.91 GiB |
| All experts at 2 bits (stand-in; the real 2.08 quant scores 0.114) | 7.137 | 0.259 | 0 |
| Least-used 25% at 1 bit, rest at 2 bits | 7.419 | 0.303 | 0.96 GiB |
| Least-used 50% at 1 bit, rest at 2 bits | 8.378 | 0.428 | 1.91 GiB |

- Moving the least-used 25% to 1 bit adds about 0.04 to 0.05 KLD and saves about 1 GiB. For the same saving, that is
  cheaper than disabling them (0.76 GiB for +0.012 at 10%; 1.9 GiB for +0.066 at 25% on the GGUF).
- Not tested with a real 1-bit trellis conversion; that needs the converter's MoE path.

## Summary of levers for a quant that fits

| Lever | Saving | Quality cost (KLD) | Evidence | What it needs |
|---|---|---|---|---|
| Remove vision tower | 0.83 GiB on disk, 0 in MLXL3 memory | 0 for text | measured | nothing; `strip_tensors.py` |
| 8-bit embeddings | 0.47 GiB | 0.005 | simulated | format or runtime support |
| Disable least-used 10% of experts | 0.76 GiB | 0.012 | measured on real files | a runtime that accepts removed experts |
| Least-used 25% of experts at 1 bit | about 1 GiB | about 0.04 to 0.05 | simulated, pessimistic | a real 1-bit conversion of those experts |

## 10. Track A: whole layers' experts at 1 bit (`layer_sensitivity.py`, simulated, stand-in quantizer)

This is the only expert lever today's runtimes can load, since both reject mixed bit widths inside a layer.
Each layer's experts are worth about 0.096 GiB when moved from 2 bits to 1.

Single layer at 1 bit, everything else untouched: KLD ranges from 0.014 (layer 23) to 0.053 (layer 39), median 0.024.
Least sensitive: layers 22 to 29, 17 and 19. Most sensitive: the last five layers (35 to 39), plus 2 and 9.

| Layers at 1 bit (least sensitive first) | KLD vs bf16 | PPL | Saving |
|---|---|---|---|
| 5 | 0.066 | 6.145 | 0.48 GiB |
| 10 | 0.136 | 6.413 | 0.96 GiB |
| 15 | 0.216 | 6.768 | 1.43 GiB |
| 20 | 0.320 | 7.387 | 1.91 GiB |

- No layer is cheap: the spread between best and worst is under 4x, and costs add almost linearly.
- For the same 1 GiB, choosing by layer costs KLD 0.136 where choosing the least-used experts across all layers cost 0.053.
  Allocation by expert usage is about 2.6 times more efficient than allocation by layer.
- The run was killed before the last check (the same sets on a 2-bit background): my script held two copies of the
  67 GiB model at once. The sweep and the table above completed and are saved.

## 11. Converter check: calibration volume and variety (`ponyexl3-convert`, MiniCPM5-1B at 4.0 bpw)

KLD against bf16, lower is better. WikiText-2 test is the standard slice; the other three texts were in none of our
calibration sets (PonyExl3 Python source, MLXL3 Rust source, MLXL3 French documentation), scored with `kld_on_text.py`.

| Quant | Calibration | WikiText-2 | Python | Rust | French docs | Convert time | Peak memory |
|---|---|---|---|---|---|---|---|
| Default (Stage 1) | 250 rows = 1 sequence, WikiText train | 0.058 | 0.045 | 0.063 | 0.065 | 4.5 min | 13 GiB |
| More volume | 16,384 rows, WikiText train | 0.027 | 0.038 | 0.049 | 0.046 | 5 to 30 min | not recorded |
| Other domain | 16,384 rows, code and instruction prompts | 0.043 | 0.020 | not run | not run | 5 min | not recorded |
| Volume and variety | 32,768 rows, prompts + WikiText + Python stdlib interleaved | 0.032 | 0.016 | 0.025 | 0.032 | 6.5 min | 108.5 GiB |
| turboderp (published, exllamav3) | unknown | 0.032 | 0.023 | 0.027 | 0.031 | n/a | n/a |

- With 32,768 rows of interleaved, varied text, `ponyexl3-convert` matches turboderp's quant on all four texts
  (better on two, level on two). The Stage 1 conclusion that the converter is "twice as lossy" was a calibration
  default, not the converter.
- A quant is good where its calibration text was: WikiText-only calibration wins on WikiText and loses elsewhere.
- The converter reads the calibration file from the start, so a mixed corpus must be interleaved, not concatenated.
- Memory is the limit: 65,536 rows peaked at 164.7 GiB and was killed twice (on the output head); 32,768 rows peaked at
  108.5 GiB. Three-quarter size would not fit in 128 GB, so 32,768 is the practical ceiling for a whole-model run here.
- Temporary disk: about 11 GB per 16,384 rows for this 1 B model.

## 12. Converter check on the MoE path (layer 23 experts, 2 bits, `ponyexl3-convert-advanced`)

- `ponyexl3-convert` (one command) cannot convert this model's routed experts: the source stores each layer's experts
  stacked in one tensor and module discovery skips them; its calibration capture cannot see them either.
- `ponyexl3-convert-advanced` can, with `--oracle-dir <published quant>` naming the modules and calibration inputs supplied
  through `--calibration-activations-map` (captured by `eval/capture_expert_acts.py`). About 2 s per expert matrix.
- 1 bit is not possible with it: `Metal trellis search currently supports K in [2, 8]`.

Output error on held-out activations (WikiText-2 test, Rust, French docs; 1,024 rows per expert; lower is better),
experts 0 and 1 of layer 23, calibrated on 4,096 rows of the interleaved corpus. Scored with `compare_expert_quants.py`.

| Quant | gate | up | down | Weight error (gate) |
|---|---|---|---|---|
| turboderp 2.08 (published) | 0.185 | 0.225 | 0.261 | 0.285 |
| ours, no Hessian shrinkage | 0.192 | 0.233 | 0.266 | 0.325 |
| ours, shrinkage 0.3 | 0.174 | 0.211 | 0.259 | 0.280 |
| ours, shrinkage 0.6 | 0.173 | 0.210 | 0.257 | 0.268 |
| ours, shrinkage 1.0 (diagonal only) | 0.197 | 0.233 | 0.262 | 0.262 |

- Without regularization our experts overfit the small calibration sample (better than turboderp's in-sample, 3 to 9% worse
  held-out on all eight experts). With `--hessian-shrinkage 0.6` they are 6% better than turboderp's on gate and up and 1%
  better on down. Two experts of one layer: an indication, not a proven result.
- Routed experts see about 3% of tokens each, so 4,096 rows per expert took 528,000 tokens of text.

## 13. Physically pruned quant in the stock engine (`prune_experts.py`)

MLXL3 reads the expert count from `config.json` and applies it to every layer, so a checkpoint in which every layer keeps
the same number of experts loads without any runtime change. `prune_experts.py` keeps each layer's most-used experts
(renumbered), trims the router to the matching rows, lowers `num_experts`, and optionally drops the vision tower. Kept
tensors are copied byte for byte.

`Qwen3.6-35B-A3B-exl3-2.08bpw-pruned10`: 230 of 256 experts per layer, vision tower removed. Stock MLXL3 engine 2.0.19.

| | Original 2.08 | Pruned copy |
|---|---|---|
| Weights on disk | 10.50 GiB | 8.87 GiB |
| PPL | 6.302 | 6.353 |
| KLD vs bf16 | 0.114 | 0.126 |
| Top-1 agreement with bf16 | 85.9% | 85.4% |
| Decode tok/s | 122 | 120 |
| Memory while generating (short prompt) | 10.30 GiB | 9.47 GiB |
| Peak memory at 8k context | 11.5 GiB | 10.7 GiB |

- KLD matches the router-masked version of the same pruning (0.126), as it should.
- Memory drops by about 0.8 GiB (the experts); the vision tower's 0.83 GiB comes off disk only.
- The pruned EXL3 quant is smaller than the IQ2_XS GGUF (10.89 GiB) and still better (KLD 0.126 vs 0.167).

## 14. Fork build

`tools/mlxl3-fork` (branch `fit-levers`, commit 1d0a519, unmodified) builds with Rust 1.100 in 65 s against MLX 0.32.2:
`cargo build --release --locked --no-default-features --features mlx,chat --bin mlxl3-rs` with `MLXL3_MLX_ROOT` set.
Packaged with its libraries in `tools/mlxl3-fork-engine`. On the original 2.08 quant it reproduces the released engine
exactly (2-window PPL 6.2656669322373855 in both; 122 tok/s decode).

## 15. Task gate: pruned vs original 2.08, IQ2_XS GGUF, and bf16

Scripts: `eval/task_gate.py` (HumanEval in `sandbox-exec`, planted codes, MMLU sample) and `eval/gate_long.py`
(16 planted codes + 4 synthetic functions to copy exactly, in 32k- and 64k-token documents). Thinking left on.
Per-item results: `results/raw/gate__<label>.json`, `results/raw/gatelong__<label>.json`.

HumanEval, 164 problems, failed items retried once with a 12,288-token reply limit:

| Model | Decoding | Passed | Stuck in thinking at the limit | Wrong |
|---|---|---|---|---|
| bf16 (mlx-lm) | greedy | 159 | 0 | 5 |
| EXL3 2.08 pruned10, 8.87 GiB (MLXL3) | greedy | 159 | 2 | 3 |
| EXL3 2.08 original, 10.50 GiB (MLXL3) | greedy | 132 | 31 | 1 |
| IQ2_XS GGUF, 10.89 GiB (llama.cpp) | greedy | 149 | 11 | 4 |
| EXL3 2.08 pruned10 | temperature 1.0, top-k 20 | 159 | 1 | 4 |
| EXL3 2.08 original | temperature 1.0, top-k 20 | 146 | 15 | 3 |

- The stuck items are real loops: the model redrafts the same correct function over and over inside its thinking.
- bf16 never loops, so this is 2-bit damage that PPL and KLD did not show (KLD ranks original 0.114 ahead of pruned 0.126).
- The GGUF and the original EXL3 quant loop on mostly different problems (5 in common).
- Why the pruned quant loops so much less is not explained. Untested guess: the rarely used experts are the worst quantized.
- Repetition penalty 1.05 (greedy) on the original's failed items: 22 retried, 5 passed, 16 still stuck; stopped early. Not a fix.
- MLXL3's server accepts temperature, top_k and repetition_penalty; it rejects top_p and seed, so sampled runs are unseeded.

Other tests:

| Test | Pruned10 | Original |
|---|---|---|
| Planted codes, 6k and 14k tokens | 32/32 | 32/32 |
| Planted codes, 32k and 64k tokens | 64/64 | 64/64 |
| Exact code copy, 32k and 64k tokens | 16/16 | 14/16 (both misses are thinking loops, not wrong copies) |
| MMLU sample, 150 questions | 121 | 124 (9 pruned-only, 12 original-only correct) |

Server speed measured back to back (same prompt, 256 and 3,000 tokens, twice): original 106 to 122 tok/s, pruned 90 to 114.
Runs made while the laptop was sleeping or hot show much lower figures and should be ignored for speed.

## 16. AppWorld pilot (30 dev tasks, 10 scenarios) and the MLXL3 thinking-history setting

Setup: AppWorld main (0.2.0.dev0), simplified ReAct code agent, first 30 tasks of `dev`, temperature 1.0, top_k 20,
8,192-token reply limit, thinking on, one run per model, unseeded. Quants served by our MLXL3 fork, bf16 by mlx-lm.

| Model | Tasks passed (of 30) | Scenarios (of 10) | Mean steps | Median thinking / step | Replies cut at 8,192 |
|---|---|---|---|---|---|
| bf16 (mlx-lm) | 15 | 1 | 13.7 | 1,571 chars | 23 of 410 |
| Original 2.08 (matched prompt) | 13 | 2 | 13.9 | 1,274 chars | 6 of 418 |
| Pruned10 + 8-bit embeddings (matched prompt) | 16 | 2 | 15.6 | 1,174 chars | 2 of 467 |

Discordant tasks: bf16 vs pruned-q8 5/6, bf16 vs original 6/4, pruned-q8 vs original 6/3. None is distinguishable
from chance at this sample size. No run hit the 50-step limit.

Three harness problems found and fixed on the way (all local patches, nothing upstreamed):

1. MLXL3 hard-codes `preserve_thinking => true` (`native/src/tokenizer.rs`), which renders every earlier assistant
   turn with an empty `<think>\n\n</think>` block. AppWorld's prompt has ten demo assistant turns, so the model copied
   "think nothing". Same messages: 3,476 prompt tokens under MLXL3 vs 3,436 under mlx-lm; rendering the template with
   and without `preserve_thinking` reproduces both numbers exactly. Fork switch: `MLXL3_PRESERVE_THINKING=0`.
   With it, median thinking went from 178 to 1,174 chars and unclosed-thinking replies from 27% to 2%.
   First-round (unmatched) scores were pruned-q8 14, original 15; kept in `outputs/_preserve_thinking_run/`.
2. Replies whose thinking block is never closed come back with an empty answer. AppWorld `language_model.py` now uses
   the reasoning text as the answer in that case (all arms).
3. AppWorld's API-call log writer crashed when the model passed a literal `...` as an argument (`requester.py`).

Open: bf16 hit the 8,192-token limit on 23 replies, far more than either quant; not yet read through. mlx-lm does
read `top_k` from the request body (`mlx_lm/server.py`), so sampling settings match across arms.

## 17. AppWorld test_normal (168 tasks), round one: pruned + 8-bit embeddings vs bf16

Same setup as section 16 (matched prompt, temperature 1.0, top_k 20, 8,192-token reply limit, earlier thinking dropped,
unseeded, one run each). AppWorld commit 42b5bcf.

| Model | Tasks passed | Difficulty 1 / 2 / 3 | Scenarios | Replies cut at 8,192 | Mean steps |
|---|---|---|---|---|---|
| bf16 (mlx-lm) | 121 of 168 (72.0%) | 50 / 33 / 38 | 44.6% | 125 of 2,500 | 14.9 |
| Pruned10 + 8-bit embeddings (MLXL3 fork) | 118 of 168 (70.2%) | 49 / 36 / 33 | 46.4% | 12 of 2,885 | 17.2 |

Discordant tasks: 24 bf16-only, 21 pruned-only (both pass 97, neither 26); exact sign test p = 0.77.

The reply limit hurt bf16: 84 of its tasks had at least one cut-off reply and it passed 54 of those (64%), against
67 of the other 84 (80%). The pruned model passed 59 and 59 on the same two groups.

There is no outside score for Qwen3.6-35B-A3B on this split in this harness to compare with. Round two (section 18)
keeps earlier thinking and raises the reply limit for both arms.

MLXL3 server speed decays within a session (63 to 40 tok/s in ten minutes, lower over hours); the pruned run was
recycled every 50 minutes. Cause not yet isolated.

## 18. AppWorld test_normal, round two: earlier thinking kept, 24,576-token reply limit

Changes from section 17, applied to both arms: `drop_reasoning_content: false`, `preserve_thinking` on (MLXL3 default
plus our fork's `reasoning_content` pass-through; mlx-lm via `chat_template_kwargs`), `max_completion_tokens` 24,576,
MLXL3 window 131,072. Prompt identity checked on a recorded 39-message conversation with 8 thinking turns: 37,918
prompt tokens on both servers and in the reference template (9,656 without `preserve_thinking`). Still top_k 20 only,
unseeded, one run each.

| Model | Round one | Round two | Difficulty 1 / 2 / 3 (r2) | Cut-off replies (r2) | Hit 50 steps (r2) | Mean steps (r2) |
|---|---|---|---|---|---|---|
| bf16 (mlx-lm) | 121 (72.0%) | 123 (73.2%) | 52 / 34 / 37 | 2 of 2,790 | 2 | 16.6 |
| Pruned10 + 8-bit embeddings | 118 (70.2%) | 110 (65.5%) | 45 / 28 / 37 | 0 of 3,353 | 7 | 20.0 |

Round two discordant tasks: 28 bf16-only, 15 pruned-only, exact sign test p = 0.066. Pooled over both rounds:
bf16 244 of 336 task-runs (72.6%), pruned 228 (67.9%).

Run-to-run noise, same model, same tasks: bf16 round one vs two flipped 36 tasks (17 / 19); pruned flipped 50
(29 / 21). A single run cannot resolve a difference of under about 10 tasks.

All 7 pruned tasks that hit 50 steps failed. One bf16 prompt reached 141,376 tokens (task 2d9f728_2; both arms passed
it); no other prompt in either arm exceeded 65,536.

No outside reference for Qwen3.6-35B-A3B on this split in this harness. How much of our 73% is the model and how much
the setup (no top-p / min-p, no `reasoning_effort`, unseeded, different server) is untested.
