function results = analyse_capture(path, varargin)
% ANALYSE_CAPTURE  First look at a raw IQ capture: level, noise floor, band occupancy.
%
%   results = ANALYSE_CAPTURE('capture.cs8')
%   results = ANALYSE_CAPTURE('capture.cs8', 'centre_hz', 446593750, 'format', 'cs8')
%
%   Parameters:
%     centre_hz       Receiver centre frequency the capture was taken at.
%     sample_rate_hz  Sample rate of the capture.
%     format          'cs8' for hackrf_transfer, 'cs16' for the PortaPack, 'cf32'.
%     fft_size        Survey transform length.
%     max_seconds     Read at most this much of the file.
%     block_seconds   Seconds read per block. Bounds memory whatever the capture length.
%
%   What this is for
%   ----------------
%   Before running anything through the node, three questions decide whether a capture is
%   worth analysing at all, and all three are answered by looking at the raw samples:
%
%     1. Did the front end saturate? If the peak sample is near full scale, every power
%        figure downstream is fiction and the capture has to be retaken with less gain or
%        more separation.
%     2. Where is the noise floor? Comparing it against a capture taken with the antenna
%        disconnected says whether the receiver is limited by its own noise or by the
%        environment, which is what decides whether an external LNA would help.
%     3. What is actually on the band? A channel that already has traffic on it is a bad
%        place to run a controlled transmission test.
%
%   The Python pipeline answers richer questions, but it takes a configured node to do it.
%   This takes a filename.
%
%   Sample scaling is the one thing worth being careful about: cs8 is signed 8-bit and cs16
%   signed 16-bit, and dividing by the wrong full scale shifts every level by tens of dB
%   without anything failing.
%
%   See also LINK_BUDGET, MEASUREMENT_SETUP, CHANNEL_PLAN.

    p = parse_options(varargin, struct( ...
        'centre_hz',      446593750, ...
        'sample_rate_hz', 2000000, ...
        'format',         'cs8', ...
        'fft_size',       1024, ...
        'max_seconds',    120, ...
        'block_seconds',  2, ...
        'verbose',        true));

    % Read and reduce in blocks. Loading a whole capture first is what makes this fall over:
    % 90 seconds at 2 MS/s is 180 million complex samples, and as doubles that alone is
    % 2.9 GB before any transform has run. The answers wanted here -- peak level and an
    % averaged spectrum -- are both reductions, and a reduction does not need its inputs kept.
    [peak, acc, n_read] = scan_file(path, p.format, p.sample_rate_hz, p.max_seconds, ...
                                    p.fft_size, p.block_seconds);
    if n_read == 0
        error('analyse_capture:empty', 'no samples read from %s', path);
    end

    peak_dbfs = 20 * log10(max(peak, 1e-12));
    saturated = peak > 0.9;

    power_db = 10 * log10(fftshift(acc) + 1e-30);
    freqs_hz = p.centre_hz + ((-p.fft_size/2):(p.fft_size/2 - 1)) * p.sample_rate_hz / p.fft_size;
    noise_floor_db = median(power_db);

    channels   = (1:16)';
    ch_freqs   = 446006250 + (channels - 1) * 12500;
    ch_over_db = zeros(16, 1);
    for k = 1:16
        [~, idx] = min(abs(freqs_hz - ch_freqs(k)));
        lo = max(1, idx - 4); hi = min(numel(power_db), idx + 4);
        ch_over_db(k) = max(power_db(lo:hi)) - noise_floor_db;
    end

    [~, peak_idx]  = max(power_db);
    peak_freq_hz   = freqs_hz(peak_idx);
    dc_offset_hz   = peak_freq_hz - p.centre_hz;

    results = struct( ...
        'seconds',        n_read / p.sample_rate_hz, ...
        'peak',           peak, ...
        'peak_dbfs',      peak_dbfs, ...
        'saturated',      saturated, ...
        'noise_floor_db', noise_floor_db, ...
        'peak_freq_hz',   peak_freq_hz, ...
        'peak_is_dc',     abs(dc_offset_hz) < p.sample_rate_hz / p.fft_size, ...
        'channel_over_db', ch_over_db, ...
        'frequencies_hz', freqs_hz, ...
        'power_db',       power_db);

    if p.verbose
        fprintf('\n=== %s ===\n\n', path);
        fprintf('duration                    %8.1f s\n', results.seconds);
        fprintf('centre                      %12.6f MHz\n', p.centre_hz / 1e6);
        fprintf('peak sample                 %8.3f    %+.1f dBFS   %s\n', peak, peak_dbfs, ...
                merge_str(saturated, 'SATURATED -- retake with less gain', 'headroom ok'));
        fprintf('noise floor                 %+8.1f dBFS\n', noise_floor_db);
        fprintf('strongest bin               %12.6f MHz  %+.1f dB over floor\n', ...
                peak_freq_hz / 1e6, max(power_db) - noise_floor_db);
        if results.peak_is_dc
            fprintf('                            ^ this is the centre frequency: LO leakage,\n');
            fprintf('                              not a signal. Retune to confirm.\n');
        end

        fprintf('\nPMR446 channels, dB over the noise floor:\n');
        for k = 1:16
            bar = repmat('#', 1, max(0, min(45, round(ch_over_db(k)))));
            if ch_over_db(k) > 10
                mark = '  <-- ACTIVITY';
            else
                mark = '';
            end
            fprintf('  PMR%-3d %+6.1f dB %s%s\n', k, ch_over_db(k), bar, mark);
        end
    end
end


function [peak, acc, n_total] = scan_file(path, format, sample_rate_hz, max_seconds, ...
                                         fft_size, block_seconds)
% Stream the capture, accumulating the peak sample and an averaged periodogram.
%
% Memory stays at one block whatever the file length, which is the whole point: the previous
% version read everything first and needed several gigabytes for a ninety-second recording.
    switch lower(format)
        case 'cs8',  precision = 'int8';    full_scale = 128;
        case 'cs16', precision = 'int16';   full_scale = 32768;
        case 'cf32', precision = 'single';  full_scale = 1;
        otherwise
            error('analyse_capture:format', 'unknown format "%s"', format);
    end

    win = blackman_harris(fft_size);
    win = win / sum(win);
    hop = fft_size / 2;

    fid = fopen(path, 'rb');
    if fid < 0
        error('analyse_capture:open', 'cannot open %s', path);
    end

    peak      = 0;
    acc       = zeros(1, fft_size);
    n_total   = 0;
    n_frames  = 0;
    remaining = 2 * floor(max_seconds * sample_rate_hz);
    block     = 2 * floor(block_seconds * sample_rate_hz);

    while remaining > 0
        raw = fread(fid, min(block, remaining), ['*' precision]);
        if numel(raw) < 2 * fft_size
            break;
        end
        remaining = remaining - numel(raw);

        raw = double(raw(1:2*floor(numel(raw)/2))) / full_scale;
        iq  = raw(1:2:end) + 1i * raw(2:2:end);
        n_total = n_total + numel(iq);
        peak = max(peak, max(abs(iq)));

        nfr = floor((numel(iq) - fft_size) / hop) + 1;
        for k = 1:nfr
            lo  = (k-1) * hop + 1;
            seg = iq(lo:lo+fft_size-1).' .* win;
            acc = acc + abs(fft(seg)) .^ 2;
            n_frames = n_frames + 1;
        end
    end
    fclose(fid);

    if n_frames > 0
        acc = acc / n_frames;
    end
end


function win = blackman_harris(n)
% Four-term Blackman-Harris, written out rather than called from a toolbox: Octave puts it in
% the signal package and MATLAB in the Signal Processing Toolbox, and depending on either
% would mean the script runs on one machine and not the next.
    a   = [0.35875, 0.48829, 0.14128, 0.01168];
    idx = 0:(n - 1);
    win = a(1) - a(2) * cos(2*pi*idx/(n-1)) ...
               + a(3) * cos(4*pi*idx/(n-1)) ...
               - a(4) * cos(6*pi*idx/(n-1));
end


function s = merge_str(condition, yes, no)
    if condition
        s = yes;
    else
        s = no;
    end
end


function opts = parse_options(args, defaults)
    opts = defaults;
    for k = 1:2:numel(args)
        name = args{k};
        if ~isfield(opts, name)
            error('analyse_capture:unknownOption', 'unknown option "%s"', name);
        end
        opts.(name) = args{k+1};
    end
end
