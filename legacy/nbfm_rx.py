#!/usr/bin/env python3
import sys
import time
from gnuradio import gr, blocks, analog, filter
from gnuradio.filter import firdes
from gnuradio.fft import window
import osmosdr

class NFMReceiver(gr.top_block):

    def __init__(self, freq_hz, output_file, duration_secs):
        gr.top_block.__init__(self)

        samp_rate_rf  = 2400000
        samp_rate_if  = 48000
        samp_rate_out = 16000
        decimation    = samp_rate_rf // samp_rate_if
        deviation_hz  = 2500

        self.source = osmosdr.source(args="hackrf=0")
        self.source.set_sample_rate(samp_rate_rf)
        self.source.set_center_freq(freq_hz)
        self.source.set_freq_corr(0)
        self.source.set_gain(16, 0)
        self.source.set_if_gain(12, 0)
        self.source.set_bb_gain(0, 0)
        self.source.set_antenna("TX/RX", 0)
        self.source.set_bandwidth(200000, 0)

        lp_taps = firdes.low_pass(
            gain             = 1.0,
            sampling_freq    = samp_rate_rf,
            cutoff_freq      = 15000,
            transition_width = 5000,
            window           = window.WIN_HAMMING
        )
        self.lp_filter = filter.fir_filter_ccf(decimation, lp_taps)

        self.nbfm = analog.nbfm_rx(
            audio_rate = samp_rate_if,
            quad_rate  = samp_rate_if,
            tau        = 75e-6,
            max_dev    = deviation_hz
        )

        self.squelch = analog.pwr_squelch_ff(
            db    = -40,
            alpha = 0.01,
            ramp  = 10,
            gate  = True
        )

        audio_taps = firdes.band_pass(
            gain             = 1.0,
            sampling_freq    = samp_rate_if,
            low_cutoff_freq  = 600,
            high_cutoff_freq = 2800,
            transition_width = 100,
            window           = window.WIN_HAMMING
        )
        self.audio_filter = filter.fir_filter_fff(1, audio_taps)

        self.agc = analog.agc_ff(
            rate      = 1e-3,
            reference = 0.5,
            gain      = 1.0
        )

        resamp_taps = firdes.low_pass(
            gain             = 1.0,
            sampling_freq    = samp_rate_if,
            cutoff_freq      = 7000,
            transition_width = 1000,
            window           = window.WIN_HAMMING
        )
        self.resamp = filter.rational_resampler_fff(
            interpolation = samp_rate_out,
            decimation    = samp_rate_if,
            taps          = resamp_taps
        )

        self.volume = blocks.multiply_const_ff(3.0)
        self.f2s    = blocks.float_to_short(1, 32767)
        self.sink   = blocks.file_sink(gr.sizeof_short, output_file, False)

        self.connect(self.source,       self.lp_filter)
        self.connect(self.lp_filter,    self.nbfm)
        self.connect(self.nbfm,         self.squelch)
        self.connect(self.squelch,      self.audio_filter)
        self.connect(self.audio_filter, self.agc)
        self.connect(self.agc,          self.resamp)
        self.connect(self.resamp,       self.volume)
        self.connect(self.volume,       self.f2s)
        self.connect(self.f2s,          self.sink)


def main():
    if len(sys.argv) < 4:
        print("uso: nbfm_rx.py <freq_hz> <output_raw> <duration_secs>")
        sys.exit(1)

    freq_hz     = int(sys.argv[1])
    output_file = sys.argv[2]
    duration    = int(sys.argv[3])

    print(f"[NFM] Sintonizando {freq_hz} Hz | {duration}s", flush=True)

    tb = NFMReceiver(freq_hz, output_file, duration)
    tb.start()
    time.sleep(duration)
    tb.stop()
    tb.wait()

    print(f"[NFM] OK: {output_file}", flush=True)


if __name__ == "__main__":
    main()
