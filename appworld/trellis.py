# Local models for the llm-trellis comparison (Qwen3.6-35B-A3B: full bf16 vs 2-bit EXL3 quants).
# Servers are started by AppWorld on a free port. MLXL3 rejects seed, stop and top_p, so none are set.
import os

ROOT = os.path.expanduser("~/llm-trellis")
MODELS = f"{ROOT}/models"
FORK = f"{ROOT}/tools/mlxl3-fork-engine-aw"
CTX = 65536
ZERO = {"input_cache_miss": 0.0, "input_cache_hit": 0.0, "input_cache_write": 0.0, "output": 0.0}


def entry(name, path, command, health):
    return {
        "model_name": name,
        "client_name": "openai",
        "model_id": path,  # MLXL3 requires the served model's absolute path as the model id
        "model_kwargs": {
            "api_type": "chat_completions",
            "temperature": 1.0,
            "api_key_env_name": "NO_API_KEY",
            "base_url": "{MODEL_SERVER_URL}/v1",
            "max_completion_tokens": 8192,
            # Same for every arm: MLXL3 ignores earlier-turn reasoning while mlx-lm would feed it back, so drop it;
            # top_k 20 is the model's shipped setting (top_p 0.95 is left out because MLXL3 rejects top_p).
            "drop_reasoning_content": True,
            "extra_body": {"top_k": 20},
            "tool_parser_name": None,
            "parallel_tool_calls": None,
            "cost_per_token": dict(ZERO),
        },
        "function_calling": False,
        "tool_choice": "auto",
        "function_calling_demos": False,
        "model_server_config": {"enabled": True, "command": command, "health_check_at": health, "timeout": 900, "show_logs": False},
        "part_of": ["vn", "vllm"],
        "provider": "vllm",
    }


def mlxl3(name, folder):
    path = f"{MODELS}/{folder}"
    cmd = f"{FORK}/mlxl3 --registry {FORK}/registry.json serve {path} --port {{port}} --context-length {CTX}"
    return entry(name, path, cmd, "/ready")


BF16 = f"{MODELS}/Qwen3.6-35B-A3B-bf16"
MODEL_INFOS = [
    entry("qwen3.6-35b-a3b-bf16", BF16, f"{ROOT}/tools/PonyExl3/.venv/bin/python -m mlx_lm server --model {BF16} --port {{port}}", "/v1/models"),
    mlxl3("qwen3.6-35b-a3b-exl3-2.08", "Qwen3.6-35B-A3B-exl3-2.08bpw"),
    mlxl3("qwen3.6-35b-a3b-exl3-2.08-pruned10-q8emb", "Qwen3.6-35B-A3B-exl3-2.08bpw-pruned10-q8emb"),
]
