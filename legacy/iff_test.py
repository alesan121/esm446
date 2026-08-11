import sys
import struct
import math

SAMPLE_RATE  = 12000
TARGET_FREQ  = 114.8
BLOCK_SIZE   = 12000
NEIGHBORS    = [90.0, 100.0, 130.0, 150.0, 170.0]
CONTRAST_THR = 3.0
VOTE_RATIO   = 0.6

def goertzel(samples, freq, rate):
    k    = round(len(samples) * freq / rate)
    w    = 2.0 * math.pi * k / len(samples)
    cr   = math.cos(w)
    ci   = math.sin(w)
    c    = 2.0 * cr
    q0 = q1 = q2 = 0.0
    for s in samples:
        q0 = c * q1 - q2 + s
        q2 = q1
        q1 = q0
    real = q1 - q2 * cr
    imag = q2 * ci
    return math.sqrt(real*real + imag*imag)

blocks_total    = 0
blocks_detected = 0

while True:
    raw = sys.stdin.buffer.read(BLOCK_SIZE * 2)
    if len(raw) < BLOCK_SIZE * 2:
        break
    samples = [struct.unpack_from("<h", raw, i*2)[0] for i in range(BLOCK_SIZE)]

    target_power   = goertzel(samples, TARGET_FREQ, SAMPLE_RATE)
    neighbor_power = sum(goertzel(samples, f, SAMPLE_RATE) for f in NEIGHBORS) / len(NEIGHBORS)
    contrast       = target_power / (neighbor_power + 1e-10)

    blocks_total    += 1
    detected         = contrast > CONTRAST_THR
    blocks_detected += int(detected)

    print(f"Target: {target_power:.0f} | Neighbors: {neighbor_power:.0f} | Contrast: {contrast:.2f}x | {'CTCSS' if detected else '----'} | Votos: {blocks_detected}/{blocks_total}", flush=True)

if blocks_total > 0:
    vote   = blocks_detected / blocks_total
    result = "ALIADO" if vote >= VOTE_RATIO else "DESCONOCIDO"
    print(f"RESULTADO: {result} ({vote*100:.0f}%)", flush=True)
