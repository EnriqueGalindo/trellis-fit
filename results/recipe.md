# Recipe: from a source model to an evaluated EXL3 quant on Apple Silicon

Parameters used throughout:

```bash
export ROOT=~/llm-trellis
export MODEL_ID=openbmb/MiniCPM5-1B        # Hugging Face source repo (bf16)
export NAME=MiniCPM5-1B                    # short name for folders and the KLD reference
export BPW=4.0                             # target bits per weight
export BUDGET_GIB=10.5                     # weight budget on the deployment machine
export PY=$ROOT/tools/PonyExl3/.venv/bin/python
```

No Homebrew, no sudo, everything under `$HOME`.

## 0. One-time setup

```bash
mkdir -p $ROOT/{tools,models,eval,results} && cd $ROOT/tools

# Python + Hugging Face CLI (log in: unauthenticated downloads are rate-limited and stall)
uv python install 3.14
uv tool install "huggingface_hub[cli]"
hf auth login

# PonyExl3: EXL3 conversion + inference. Clone from beamivalice, not the URL in its README.
git clone https://github.com/beamivalice/PonyExl3 && cd PonyExl3
uv venv --python 3.14 --managed-python .venv
uv pip install --python .venv/bin/python -e ".[dev]" && cd ..

# MLXL3 engine (macOS 26.2+): the standalone engine tarball, no Desktop app needed
curl -fsSLO https://github.com/0xZKnw/mlxl3/releases/download/engine-v2.0.19/MLXL3-Engine-v2.0.19-arm64.tar.gz
curl -fsSL https://github.com/0xZKnw/mlxl3/releases/download/engine-v2.0.19/MLXL3-Engine-v2.0.19-arm64.tar.gz.sha256
shasum -a 256 MLXL3-Engine-v2.0.19-arm64.tar.gz          # must match the line above
mkdir -p mlxl3-engine && tar -xzf MLXL3-Engine-v2.0.19-arm64.tar.gz -C mlxl3-engine

# llama.cpp prebuilt. The "latest" release has no binaries; read the nightly tag from it.
TAG=$(curl -sL https://github.com/ggml-org/llama.cpp/releases/download/v1.0.0/nightly-tag.txt)
curl -fsSLO https://github.com/ggml-org/llama.cpp/releases/download/$TAG/llama-$TAG-bin-macos-arm64.tar.gz
mkdir -p llama.cpp && tar -xzf llama-$TAG-bin-macos-arm64.tar.gz -C llama.cpp
export LLAMA_DIR=$ROOT/tools/llama.cpp/llama-$TAG       # the eval reads this (default: llama-b12359)

# WikiText-2 test set
cd $ROOT/eval && curl -fsSLO https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip
unzip -oq wikitext-2-raw-v1.zip && cp wikitext-2-raw/wiki.test.raw .
```

The eval scripts (`run_eval.py`, `report.py`, `controls.py`, `evallib.py`, `worker_*.py`) live in `$ROOT/eval`
and always run with `$PY`.

## 1. Get the source and build the KLD reference

```bash
cd $ROOT/models
until hf download $MODEL_ID --local-dir $NAME-bf16; do sleep 30; done    # retry loop: downloads die on network changes
ls $NAME-bf16/*.safetensors | wc -l                                      # confirm every shard arrived

cd $ROOT/eval
$PY run_eval.py ../models/$NAME-bf16 mlx --save-ref $NAME                # load check, PPL, speed, memory + saves the reference
$PY controls.py $NAME ../models/$NAME-bf16 <another-reference-name>      # negative controls; must print "all_pass": true
```

The reference costs about `16 x 1023 x vocab x 2` bytes on disk (4 GB for a 130k vocab, 7.6 GB for Qwen's 248k).
Skip this step, and `--ref` below, if the bf16 source will not fit in memory.

## 2. Convert (or download) the EXL3 quant

```bash
cd $ROOT
tools/PonyExl3/.venv/bin/ponyexl3-convert \
  --in-dir models/$NAME-bf16 \
  --out-dir models/$NAME-exl3-${BPW}bpw \
  --work-dir models/_work/$NAME-$BPW \
  --bits $BPW \
  --calibration-rows 32768 \
  --calibration-text eval/calib_interleaved.txt   # add --resume after an interruption
```

Set the calibration explicitly. The default is 250 rows from one WikiText sequence and produced quants with about
twice the KLD of published ones. 32,768 rows of interleaved, varied text matched them, at 108 GiB peak memory on a
1 B model; 65,536 rows ran out of memory on a 128 GB machine.

A published exllamav3 quant, when one exists, saves the conversion (`hf download <repo> --revision <bpw-branch> --local-dir ...`).
With the calibration above, `ponyexl3-convert` output measured level with it on a 1 B dense model; the MoE path is still unverified.

## 3. Evaluate

```bash
cd $ROOT/eval
Q=../models/$NAME-exl3-${BPW}bpw
$PY run_eval.py $Q pony  --ref $NAME --budget-gib $BUDGET_GIB        # PonyExl3
$PY run_eval.py $Q mlxl3 --ref $NAME --budget-gib $BUDGET_GIB        # MLXL3
$PY run_eval.py ../models/gguf/<file>.gguf llama --ref $NAME --budget-gib $BUDGET_GIB   # GGUF baseline
```

Each run does: load check, PPL (WikiText-2, 16 x 2048-token windows, second half scored), KLD vs the reference,
speed (512-token prompt, 128 generated, greedy, median of 3), and peak memory through an 8,192-token context.
A failed load is recorded with its error and the other tests are skipped. Results land in `results/raw/`.

## 4. Report

Write a manifest (see `eval/stage3_manifest.json`) listing the rows, then:

```bash
$PY report.py my_manifest.json > ../results/my_stage.md
```

## Choosing a bitrate for a memory budget

- Budget the **weight files**, not the nominal bpw. Nominal bpw leaves out embeddings and the output head:
  "2.08 bpw" Qwen3.6-35B-A3B is 2.51 effective; "4.0 bpw" MiniCPM5-1B is 6.6 effective.
  Rough size: `params x nominal_bpw / 8`, plus the bf16 embedding table, plus the head at 6 bits.
- Leave about 1 GiB above the weights for an 8k context in MLXL3 or llama.cpp (measured: 10.50 GiB of weights
  peaked at 11.5 GiB).
- PonyExl3's load transient is about 2x the weights (115 GiB for a 59 GiB model). Call `mx.clear_cache()` after
  `load_model` (the eval does this with `EVAL_CLEAR_CACHE_AFTER_LOAD=1`). Do not plan on PonyExl3 for a model
  larger than about half the machine's memory.
- Small models do not survive 2 bpw (1 B model: KLD 0.57 to 1.2, wrong answers). A 35 B MoE does (KLD 0.11).

## Pitfalls hit, in the order they cost time

1. **Runtime support is narrower than the READMEs suggest. Run the load check first.**
   - MLXL3 has no Llama-layout runtime (`native chat does not support model type "llama"`).
   - MLXL3 needs `partial_rotary_factor` and `tie_word_embeddings: false` at the top of `text_config`; older Qwen
     EXL3 exports (Qwen3.5-2B, turboderp's Qwen3.5-122B) fail with `missing field partial_rotary_factor`.
   - PonyExl3 cannot load MoE quants whose shared expert has a different bit width than the routed experts
     (every Qwen3.6-35B-A3B quant below 4 bpw that we tried: 2.08, 2.49):
     `ValueError: [concatenate] All the input array dimensions must match`. Uniform quants (4.00) load.
2. **`ponyexl3-convert`'s default calibration is far too small** (one sequence). With defaults it scored KLD 0.058 vs 0.032 for
   exllamav3 at 4.0 bpw on MiniCPM5-1B; with 32,768 interleaved rows it scored 0.032.
   For MoE models it skips routed experts unless `--include-routed-experts` is passed, and that path is unverified.
3. **Tokenizer BOS can disagree with itself.** MiniCPM5's `tokenizer_config.json` says no BOS, its `tokenizer.json`
   adds one, and the GGUF follows the config. The eval follows `tokenizer_config.json`; a mismatch is caught by the
   token-slice hash and KLD is refused rather than computed on shifted tokens.
4. **Memory measurement.** Use process physical footprint, which includes Metal memory. With mmap, GGUF weights
   are file-backed and invisible to it, so the memory test loads with `--load-mode none` (older builds: `--no-mmap`).
5. **llama.cpp flags and builds move.** `--no-mmap` is gone in b12359; new architectures need new builds.
   Inkling-2-Mini's top-level GGUFs need Unsloth's fork (`unslothai/llama.cpp` release `b12588-mix-c0c3de8`);
   mainline wants the files under `Mainline_Support/`.
6. **Downloads.** Log in first. A network change stalls the downloader silently (process alive, files not growing),
   and a restart does not reuse partial files. Check the shard count before using a model, and never judge a
   download by the exit status of a command chain that ends in `echo`.
7. **MLXL3's CLI is chat-only.** `mlxl3 run` always applies the chat template and stops at EOS, so its speed prompt
   carries a few template tokens; `mlxl3 forward` gives raw logits and is what PPL/KLD use (about 33 to 42 s per window
   for a 35 B model, since it reloads the model per window).
8. **Upstream `mlx_eval.compare` cannot load EXL3.** KLD here uses the same formula on WikiText-2, with the F16 GGUF
   in llama.cpp scoring KLD 0.000 against bf16 in mlx-lm as the cross-runtime check.
9. **Re-measure outliers.** One speed sample read 25% low for no visible reason; repeat runs agreed with each other.

## 5. Optional: prune rarely used experts (MoE models), then task-test

```bash
cd $ROOT/eval
$PY sim_levers.py ../models/$NAME-bf16 $NAME base                 # first run writes eval/refs/$NAME/router_stats.npz (expert usage)
$PY prune_experts.py ../models/<quant> ../models/<quant>-pruned10 refs/$NAME/router_stats.npz 10 --drop-vision
$PY run_eval.py ../models/<quant>-pruned10 mlxl3 --ref $NAME --budget-gib $BUDGET_GIB
$PY task_gate.py <label> mlxl3 ../models/<quant>-pruned10            # coding, recall, multiple choice
$PY task_gate.py <label> mlxl3 ../models/<quant>-pruned10 --tasks humaneval,mmlu --retry 12288   # rerun cut-off items
$PY gate_long.py <label> mlxl3 ../models/<quant>-pruned10            # recall + exact copy at 32k and 64k tokens
```

Do not choose between quants on PPL and KLD alone. On Qwen3.6-35B-A3B the quant with the better KLD was the one
that got stuck in its thinking on 31 of 164 coding problems; only the task gate showed it.

## Final recommendation for the 16 GB mini

**Use the pruned 2.08 bpw EXL3 quant (`models/Qwen3.6-35B-A3B-exl3-2.08bpw-pruned10`, 8.87 GiB) in MLXL3.**
Keep the **IQ2_XS GGUF in llama.cpp** as the fallback. Nothing here has been run on the mini itself.

| | Pruned EXL3 2.08 | Original EXL3 2.08 | IQ2_XS GGUF |
|---|---|---|---|
| Weights | 8.87 GiB | 10.50 GiB | 10.89 GiB |
| Peak memory at 8k context | 10.7 GiB | 11.5 GiB | not comparable (file-backed with mmap) |
| PPL (bf16 5.90) / KLD | 6.35 / 0.126 | 6.30 / 0.114 | 6.62 / 0.167 |
| HumanEval, greedy (bf16 159) | 159 | 132 | 149 |
| Stuck in thinking, greedy / sampled | 2 / 1 | 31 / 15 | 11 / not run |
| Recall and exact copy to 64k tokens | all correct | 2 copies lost to thinking loops | not run |
| Decode speed | about 110 tok/s | about 120 tok/s | 117 to 136 tok/s |

- **Why the pruned quant:** it is 1.6 GiB smaller than the original, matched the bf16 model on the coding test, and
  almost never gets stuck in its thinking, which the original does on roughly one coding problem in five (greedy)
  or one in eleven (sampled). Its KLD is slightly worse; the task tests did not show any cost from that.
- **How it was made:** the original `UnstableLlama/Qwen3.6-35B-A3B-exl3-2.08bpw` with the 26 least-used experts
  removed from every layer (230 of 256 kept) and the unused vision tower dropped, by `eval/prune_experts.py`.
  It loads in the stock MLXL3 engine 2.0.19. It is text-only.
- **Why it loops less is not understood,** so re-run the task gate on any new pruned quant instead of assuming it.
- **Settings:** MLXL3's server accepts temperature, top-k and repetition penalty, not top-p or seed. Use
  temperature 1.0 with top-k 20. A repetition penalty of 1.05 did not stop the loops.
- **Memory is still the open risk.** 10.7 GiB peak at 8k context has to fit under whatever macOS lets the GPU wire on
  a 16 GB machine, which I did not measure. In the simulated-pressure test the original EXL3 quant dropped to
  37 tok/s with 8.5 GiB available while the GGUF held full speed, because llama.cpp's weights are file-backed and
  can be paged. The pruned quant was not run under that test.
- **Fallback, IQ2_XS GGUF in llama.cpp (default mmap mode):** tolerates memory pressure best, but has worse KLD and
  got stuck on 11 coding problems.
- **Second fallback, `unsloth/Inkling-2-Mini-GGUF` UD-Q2_K_XL** (8.51 GiB, 10.3 GiB peak): a different model, a third
  of the speed here (40 tok/s), needs Unsloth's llama.cpp fork, and was not task-tested.
- Not recommended: the original 2.08 quant (larger and loops), 2.49 bpw EXL3 (12.1 GiB), Inkling UD-IQ3_XXS (10.86 GiB).
- MLXL3's own validation was on M5 hardware; M1/M2 is untested by its authors and by us.

First thing to do on the mini: copy the pruned quant over and run `run_eval.py <model> mlxl3 --tests load,mem`, then
`task_gate.py <label> mlxl3 <model> --tasks needles,humaneval --limit 20`, and watch the peak memory and the speed.
