#!/usr/bin/env python3
"""Pin memory so the rest of the machine behaves like a smaller one. Usage: ballast.py <GiB> <max_seconds>

Locks <GiB> of RAM (mlock, so it cannot be compressed or swapped), prints READY, then sleeps until
killed or until max_seconds pass. Everything is released the moment the process exits.
"""
import ctypes, mmap, sys, time
gib, life = float(sys.argv[1]), float(sys.argv[2])
libc = ctypes.CDLL("/usr/lib/libSystem.dylib", use_errno=True)
libc.mlock.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
CH = 1 << 30
chunks, left = [], int(gib * CH)
while left > 0:
    n = min(CH, left)
    m = mmap.mmap(-1, n)
    addr = ctypes.addressof(ctypes.c_char.from_buffer(m))
    if libc.mlock(addr, n) != 0:
        print(f"MLOCK_FAILED errno={ctypes.get_errno()} after {len(chunks)} GiB", flush=True); sys.exit(2)
    chunks.append(m); left -= n
print(f"READY {len(chunks)} chunks", flush=True)
time.sleep(life)
