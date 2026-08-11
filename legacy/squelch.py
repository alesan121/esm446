import sys
import struct
import math

SAMPLE_RATE   = 48000
BLOCK_SAMPLES = 4800
SQUELCH_RMS   = 10000

def rms(samples):
    return math.sqrt(sum(s*s for s in samples) / len(samples))

buf = b""
active = False
silence_blocks = 0
MAX_SILENCE = 3

while True:
    chunk = sys.stdin.buffer.read(BLOCK_SAMPLES * 2)
    if not chunk:
        break
    samples = [struct.unpack_from("<h", chunk, i*2)[0] for i in range(len(chunk)//2)]
    level = rms(samples)
    if level > SQUELCH_RMS:
        active = True
        silence_blocks = 0
        sys.stdout.buffer.write(chunk)
        sys.stdout.buffer.flush()
    elif active:
        silence_blocks += 1
        sys.stdout.buffer.write(chunk)
        sys.stdout.buffer.flush()
        if silence_blocks >= MAX_SILENCE:
            active = False
            silence_blocks = 0
