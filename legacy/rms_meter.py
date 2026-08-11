import sys, struct, math

BLOCK = 4800
while True:
    raw = sys.stdin.buffer.read(BLOCK * 2)
    if len(raw) < BLOCK * 2:
        break
    samples = [struct.unpack_from("<h", raw, i*2)[0] for i in range(BLOCK)]
    rms = math.sqrt(sum(s*s for s in samples) / len(samples))
    print(f"RMS: {rms:.1f}", flush=True)
