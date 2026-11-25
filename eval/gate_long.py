#!/usr/bin/env python3
"""Long-context damage detector (after Cary Palmer's Kintsugi recall tests).

  gate_long.py <label> <runtime> <model_path> [--sizes 32k,64k] [--docs 2]

Per document (WikiText filler of ~32k or ~64k tokens):
  needles  16 planted 6-digit codes, all asked for in one reply
  copy     4 planted synthetic Python functions (random names and constants, so nothing can be recalled from
           training); each is asked for separately and must come back character for character

Results per item go to results/raw/gatelong__<label>.json; reruns resume.
"""
from __future__ import annotations

import argparse, json, random, re, time
from pathlib import Path

import evallib as E
import task_gate as G

CHARS = {"8k": 33000, "16k": 66000, "32k": 133000, "64k": 266000}
CTX_FOR = {"8k": 12288, "16k": 20480, "32k": 40960, "64k": 73728}
NAMES = ["amber", "birch", "cobalt", "delta", "ember", "fjord", "garnet", "harbor",
         "indigo", "juniper", "kestrel", "lagoon", "marble", "nectar", "onyx", "pewter"]
WORDS = ["tally", "merge", "shift", "probe", "blend", "fold", "trim", "rank", "scan", "pack", "split", "clamp"]
SNIPS = ["alpha", "bravo", "charlie", "dexter"]


def make_func(rnd: random.Random) -> str:
    f = f"{rnd.choice(WORDS)}_{rnd.choice(WORDS)}_{rnd.randrange(100, 999)}"
    a, b, c = (rnd.choice("abcdefghkmnpqrstuvwxyz") + str(rnd.randrange(2, 99)) for _ in range(3))
    k = [rnd.randrange(3, 9973) for _ in range(5)]
    op = [rnd.choice(["+", "-", "*", "^", "%"]) for _ in range(3)]
    return (f"def {f}({a}, {b}={k[0]}):\n"
            f"    {c} = [({a} {op[0]} i) {op[1]} {k[1]} for i in range({b} % {k[2] % 37 + 5})]\n"
            f"    if sum({c}) % {k[3] % 11 + 2} == {k[4] % 2}:\n"
            f"        return sorted({c})[: {k[2] % 9 + 1}]\n"
            f"    return [x {op[2]} {k[4]} for x in {c} if x != {k[3]}]")


def norm(code: str) -> str:
    return "\n".join(l.rstrip() for l in code.strip().splitlines())


def build(size: str, doc: int):
    rnd = random.Random(f"long-{size}-{doc}")
    text = E.WIKI.read_text(encoding="utf-8")
    start = rnd.randrange(0, len(text) - CHARS[size] - 1)
    paras = text[start : start + CHARS[size]].split("\n")
    codes = {n: f"{rnd.randrange(100000, 999999)}" for n in NAMES}
    funcs = {s: make_func(rnd) for s in SNIPS}
    inserts = [f" The secret code for project {n} is {c}. " for n, c in codes.items()]
    inserts += [f"\nSnippet {s}:\n```python\n{fn}\n```\n" for s, fn in funcs.items()]
    rnd.shuffle(inserts)
    slots = sorted(rnd.sample(range(1, len(paras) - 1), len(inserts)))
    for ins, s in zip(inserts, reversed(slots)):
        paras.insert(s, ins)
    return "<document>\n" + "\n".join(paras) + "\n</document>\n\n", codes, funcs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("label"); ap.add_argument("runtime"); ap.add_argument("model")
    ap.add_argument("--sizes", default="32k,64k"); ap.add_argument("--docs", type=int, default=2)
    a = ap.parse_args()
    sizes = a.sizes.split(",")
    out = E.RAW_DIR / f"gatelong__{a.label}.json"
    doc = json.loads(out.read_text()) if out.exists() else {"label": a.label, "runtime": a.runtime, "model": a.model, "items": {}}
    items = doc["items"]
    G.CTX = max(CTX_FOR[s] for s in sizes)
    srv = G.Server(a.runtime, str(Path(a.model).resolve()), "long_" + a.label)
    t0 = time.time()
    try:
        for size in sizes:
            for d in range(a.docs):
                body, codes, funcs = build(size, d)
                head = "Read the document, then answer the question after it.\n\n" + body
                key = f"{size}-{d}:needles"
                if key not in items:
                    r = srv.chat(head + "The document contains secret codes for these projects: " + ", ".join(NAMES) +
                                 ". Reply with one line per project in the form `name: code`.", 3072)
                    wrong = [n for n, c in codes.items() if not re.search(rf"{n}\W+{c}\b", r["answer"])]
                    items[key] = {"found": len(NAMES) - len(wrong), "of": len(NAMES), "wrong": wrong,
                                  "seconds": round(r["seconds"], 1), "why": r.get("error") or r.get("finish")}
                    out.write_text(json.dumps(doc, indent=1))
                for s, fn in funcs.items():
                    key = f"{size}-{d}:copy-{s}"
                    if key in items:
                        continue
                    r = srv.chat(head + f"The document contains a Python code block labelled `Snippet {s}`. Reproduce that function exactly, "
                                 "character for character, in a single ```python code block and nothing else.", 3072)
                    got = norm(G.extract_code(r["answer"]))
                    items[key] = {"ok": got == norm(fn), "seconds": round(r["seconds"], 1), "why": r.get("error") or r.get("finish"),
                                  **({} if got == norm(fn) else {"got": got[:600], "want": fn})}
                    out.write_text(json.dumps(doc, indent=1))
                nd = [v for k, v in items.items() if k.startswith(size) and "needles" in k]
                cp = [v for k, v in items.items() if k.startswith(size) and "copy" in k]
                print(json.dumps({"label": a.label, "size": size, "doc": d, "needles": f"{sum(v['found'] for v in nd)}/{sum(v['of'] for v in nd)}",
                                  "copy": f"{sum(v['ok'] for v in cp)}/{len(cp)}", "minutes": round((time.time() - t0) / 60, 1)}), flush=True)
    finally:
        srv.stop()
        out.write_text(json.dumps(doc, indent=1))


if __name__ == "__main__":
    main()
