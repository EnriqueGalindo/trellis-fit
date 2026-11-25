"""List GGUF tensor names, shapes, types and sizes matching a regex. Usage: gguf_tensors.py <file.gguf> [regex]"""
import re, sys
from collections import Counter
from gguf import GGUFReader
r = GGUFReader(sys.argv[1]); pat = re.compile(sys.argv[2] if len(sys.argv) > 2 else ".")
c = Counter()
for t in r.tensors:
    c[(re.sub(r"\d+", "N", t.name), t.tensor_type.name, tuple(int(x) for x in t.shape))] += int(t.n_bytes)
for (n, ty, sh), b in sorted(c.items(), key=lambda x: -x[1]):
    if pat.search(n): print(f"{b / 2**30:8.3f} GiB  {ty:8s} {str(sh):24s} {n}")
