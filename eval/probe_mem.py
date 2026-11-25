"""Break down process footprint vs MLX accounting right after an EXL3 load. Usage: probe_mem.py <model_dir>"""
import sys, json, mlx.core as mx, evallib as E
from ponyexl3.mlx.model import load_model
m, _ = load_model(sys.argv[1], engine="exl3", warm=True)
def snap(tag):
    f = E.footprint()
    print(json.dumps({"at": tag, "footprint_gib": round(f["footprint_gib"], 2), "peak_gib": round(f["peak_footprint_gib"], 2),
                      "mlx_active_gib": round(mx.get_active_memory() / E.GIB, 2), "mlx_cache_gib": round(mx.get_cache_memory() / E.GIB, 2)}))
snap("after load"); mx.clear_cache(); snap("after clear_cache")
