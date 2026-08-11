#!/usr/bin/env python3
import sys
import time
import numpy as np
from scipy import signal as dsp
import SoapySDR
from SoapySDR import SOAPY_SDR_RX, SOAPY_SDR_CF32

# Centro de los 16 canales PMR446 EU
# Canal 1: 446.00625 -- Canal 16: 446.18125
# Centro exacto: 446.09375 MHz
CENTER_FREQ  = 446093500
SAMP_RATE    = 800000       # ampliado: cubre 445.7-446.5 MHz con margen
GAIN_LNA     = 0
GAIN_VGA     = 20
BLOCK_SIZE   = 262144
AUDIO_RATE   = 12000
SNR_FACTOR   = 2.0
MIN_POWER    = 0.065

# Grid dinamico alineado al origen PMR1 (446.00625 MHz), paso 12.5 kHz NFM.
# Cubre toda la banda capturada -- detecta canales nominales Y frecuencias atipicas.
# Analogia: en vez de 16 filtros de cristal fijos, generamos N filtros sinteticos
# uniformemente distribuidos sobre el BW del receptor.
_STEP   = 12500
_MARGIN = int(SAMP_RATE * 0.45)          # margen del 10% en cada borde
_F_MIN  = CENTER_FREQ - _MARGIN
_F_MAX  = CENTER_FREQ + _MARGIN
_ORIGIN = 446006250                       # PMR canal 1 exacto
_start  = _ORIGIN - (((_ORIGIN - _F_MIN) // _STEP) + 1) * _STEP
CHANNELS = {
    i: f
    for i, f in enumerate(range(_start, _F_MAX, _STEP), start=1)
    if f >= _F_MIN
}

# Mapa inverso: frecuencia → numero de canal PMR nominal (1-16) o 0 si atipico
_PMR_NOMINAL = {446006250 + n * 12500: n + 1 for n in range(16)}
def _pmr_ch(freq_hz):
    return _PMR_NOMINAL.get(freq_hz, 0)

def get_channel_iq(iq, ch_freq, center, samp_rate, audio_rate):
    offset  = ch_freq - center
    t       = np.arange(len(iq)) / samp_rate
    shifted = iq * np.exp(-2j * np.pi * offset * t).astype(np.complex64)
    dec     = int(samp_rate / audio_rate)
    lp      = dsp.firwin(101, audio_rate / samp_rate)
    filt    = dsp.lfilter(lp, 1.0, shifted)
    return filt[::dec]

def nfm_demodulate(iq):
    iq_norm = iq / (np.abs(iq) + 1e-10)
    demod   = np.angle(iq_norm[1:] * np.conj(iq_norm[:-1]))
    return demod.astype(np.float32)

def iq_power(iq):
    return float(np.mean(np.abs(iq)**2))

def main():
    if len(sys.argv) < 2:
        print("uso: channelizer.py <output_dir>")
        sys.exit(1)

    output_dir = sys.argv[1]

    # Inicializacion unica -- HackRF nunca se cierra
    sdr = SoapySDR.Device(dict(driver="hackrf"))
    sdr.setSampleRate(SOAPY_SDR_RX, 0, SAMP_RATE)
    sdr.setFrequency(SOAPY_SDR_RX, 0, CENTER_FREQ)
    sdr.setGain(SOAPY_SDR_RX, 8, "LNA", GAIN_LNA)
    sdr.setGain(SOAPY_SDR_RX, 16, "VGA", GAIN_VGA)
    sdr.setGain(SOAPY_SDR_RX, 0, "AMP", 0)
    rxStream = sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32)
    sdr.activateStream(rxStream)

    print("[WB] Daemon arrancado -- HackRF activo permanente", flush=True)

    while True:
        audio_buffers  = {ch: [] for ch in CHANNELS}
        active_channel = None
        capture_start  = None

        # Fase de deteccion -- espera senal
        while active_channel is None:
            buff = np.zeros(BLOCK_SIZE, dtype=np.complex64)
            sr   = sdr.readStream(rxStream, [buff], BLOCK_SIZE, timeoutUs=1000000)
            if sr.ret < 0:
                continue
            iq = buff[:sr.ret]

            channel_powers = {}
            channel_iqs    = {}
            for ch_num, ch_freq in CHANNELS.items():
                ch_iq                  = get_channel_iq(iq, ch_freq, CENTER_FREQ, SAMP_RATE, AUDIO_RATE)
                channel_powers[ch_num] = iq_power(ch_iq)
                channel_iqs[ch_num]    = ch_iq

            best_ch    = max(channel_powers, key=channel_powers.get)
            best_power = channel_powers[best_ch]
            noise_avg  = np.mean([p for ch, p in channel_powers.items() if ch != best_ch])

            if best_power > MIN_POWER and best_power > noise_avg * SNR_FACTOR:
                active_channel = best_ch
                capture_start  = time.time()
                # Guarda el bloque que detecto la senal -- sin perder inicio
                audio = nfm_demodulate(channel_iqs[best_ch])
                audio_buffers[best_ch].append(audio)
                print(f"[WB] Canal {best_ch} ({CHANNELS[best_ch]}Hz) pwr:{best_power:.4f}", flush=True)

        # Fase de grabacion -- graba hasta 10s o silencio
        silence_blocks = 0
        while True:
            if time.time() - capture_start > 10:
                break

            buff = np.zeros(BLOCK_SIZE, dtype=np.complex64)
            sr   = sdr.readStream(rxStream, [buff], BLOCK_SIZE, timeoutUs=1000000)
            if sr.ret < 0:
                continue
            iq = buff[:sr.ret]

            ch_iq  = get_channel_iq(iq, CHANNELS[active_channel], CENTER_FREQ, SAMP_RATE, AUDIO_RATE)
            power  = iq_power(ch_iq)

            if power > MIN_POWER:
                silence_blocks = 0
                audio = nfm_demodulate(ch_iq)
                audio_buffers[active_channel].append(audio)
            else:
                silence_blocks += 1
                if silence_blocks >= 3:
                    break

        # Guarda la captura
        if audio_buffers[active_channel]:
            ts         = time.strftime("%Y%m%d_%H%M%S")
            raw_file   = f"{output_dir}/raw_{ts}.s16"
            freq_file  = f"{output_dir}/freq_{ts}.txt"

            audio_out = np.concatenate(audio_buffers[active_channel])
            audio_int = np.clip(audio_out * 32767, -32767, 32767).astype(np.int16)
            audio_int.tofile(raw_file)

            pmr_ch = _pmr_ch(CHANNELS[active_channel])
            with open(freq_file, "w") as f:
              f.write(f"{pmr_ch},{CHANNELS[active_channel]},{best_power:.6f}")

            print(f"[WB] Guardado: canal {active_channel} | {len(audio_int)} muestras", flush=True)


if __name__ == "__main__":
    main()
