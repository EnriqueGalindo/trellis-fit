# trellis-fit

Scripts and results for fitting a 2-bit EXL3 mixture-of-experts model into a 16 GB Apple Silicon Mac by removing
rarely used experts and storing the embedding table in 8 bits.

- Write-up: [docs/index.html](docs/index.html)
- Models: [pruned10](https://huggingface.co/meatlo4f/Qwen3.6-35B-A3B-exl3-2.08bpw-pruned10) (stock MLXL3) and
  [pruned10-q8emb](https://huggingface.co/meatlo4f/Qwen3.6-35B-A3B-exl3-2.08bpw-pruned10-q8emb) (needs the fork)
- Engine fork: [EnriqueGalindo/mlxl3, branch `fit-levers`](https://github.com/EnriqueGalindo/mlxl3/tree/fit-levers)

## Rebuild the two models

You need Python 3.12 with `numpy`, and the source quant
[UnstableLlama/Qwen3.6-35B-A3B-exl3-2.08bpw](https://huggingface.co/UnstableLlama/Qwen3.6-35B-A3B-exl3-2.08bpw).

```sh
cd eval
python prune_experts.py <source_quant_dir> <out>/pruned10 refs/Qwen3.6-35B-A3B/router_stats.npz 10 --drop-vision
python embed_q8.py <out>/pruned10 <out>/pruned10-q8emb
```

Nothing is re-quantized, so the output is deterministic. Checked on 2026-11-25: both weight shards of both models
come out byte-identical to the published files.

| File | SHA-256 (first 12) |
|---|---|
| pruned10 `model-00001-of-00002.safetensors` | `c4e133fa207b` |
| pruned10-q8emb `model-00001-of-00002.safetensors` | `87e7fb765681` |
| both, `model-00002-of-00002.safetensors` | `fbc56322cfa7` |

`router_stats.npz` holds how often each expert was routed to in each layer of the bf16 model, counted on the
WikiText-2 training split plus a sample of code. `sim_levers.py ... base` regenerates it from the bf16 model; the
calibration text itself is not in this repository.

## Layout

| Path | What it is |
|---|---|
| `eval/prune_experts.py` | Removes the least-used experts from every layer of an EXL3 checkpoint |
| `eval/embed_q8.py` | Rewrites the embedding table as int8 codes plus one f16 scale per row |
| `eval/run_eval.py`, `worker_*.py`, `evallib.py`, `report.py` | Perplexity, KL against a bf16 reference, speed and memory, per runtime |
| `eval/task_gate.py` | HumanEval (sandboxed), planted-code recall, an MMLU sample |
| `eval/gate_long.py` | Recall and exact code copy at 32k and 64k tokens |
| `eval/server_speed.py` | Generation speed through the MLXL3 server |
| other `eval/*.py` | One-off experiments referred to in `results/experiments_fit.md` |
| `appworld/` | Our AppWorld model definitions and a three-part local patch (commit `42b5bcf`) |
| `results/recipe.md` | The step-by-step recipe |
| `results/experiments_fit.md` | The full lab notebook, including what did not work |
| `results/raw/` | One JSON per model and runtime, and per task test |
| `results/appworld_test_normal_pass_fail.csv` | Pass or fail per task for the four AppWorld runs |

The scripts expect this layout, with models under `../models` and engines under `../tools`, or the `MLXL3_BIN` and
`LLAMA_DIR` environment variables.

## Data that is not included

- WikiText-2 (`eval/wiki.test.raw`): from
  `https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip`.
- HumanEval (`eval/gate/HumanEval.jsonl.gz`): from the `openai/human-eval` repository.
- The bf16 reference log-probabilities (7.6 GB): `run_eval.py` writes them from the bf16 model.
- AppWorld task outputs and logs. AppWorld asks that test content is not published; only pass or fail per task id
  is given here.

## AppWorld

`appworld/trellis.py` goes in `experiments/configs/_generator/models/`. `appworld/appworld-local.patch` makes three
changes to AppWorld: it forwards `extra_body` to the model call, uses the reasoning text as the answer when a reply
never closes its thinking block, and stops the API-call log writer crashing on a literal `...` argument.

Round one used the generated configs as they are, with `MLXL3_PRESERVE_THINKING=0` set for the MLXL3 server. Round two
changed `max_completion_tokens` to 24576 and `drop_reasoning_content` to false in both arms, added
`"chat_template_kwargs": {"preserve_thinking": true}` to the mlx-lm arm's `extra_body`, left `MLXL3_PRESERVE_THINKING`
unset, and gave MLXL3 a 131072-token window.

## License

MIT for the code in this repository. Models, MLXL3, AppWorld, HumanEval, MMLU and WikiText keep their own licenses.
