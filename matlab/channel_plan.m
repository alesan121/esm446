function results = channel_plan(varargin)
% CHANNEL_PLAN  Offset tuning: grid alignment, bin mapping, DC spur and IQ images.
%
%   results = CHANNEL_PLAN() checks the shipping configuration and prints the mapping.
%
%   results = CHANNEL_PLAN('centre_hz', 446093750) checks a different centre frequency, for
%   example to see why channel 8 was rejected.
%
%   Parameters:
%     centre_hz      Receiver centre frequency.
%     sample_rate_hz Receiver sample rate.
%     num_channels   Polyphase filter bank channel count.
%
%   Two constraints, pulling in different directions
%   ------------------------------------------------
%   A polyphase channeliser produces bins at centre + k*fs/M. For the PMR446 channels to land
%   exactly on bins, the centre must be an integer number of 12.5 kHz steps from channel 1.
%   The midpoint of the allocation, 446.1 MHz, is 7.5 steps out -- half a bin -- so every
%   channel would straddle two.
%
%   That constraint alone is satisfied by every channel, and choosing one of them is a trap.
%   A direct-conversion receiver leaks its local oscillator to DC, putting a spur at its own
%   centre frequency: measured at +31 dB above the noise floor on a HackRF One. It also
%   mirrors every signal about DC through IQ imbalance. Centre the receiver inside the
%   allocation and the spur lands on a channel while the mirrors land on other channels.
%
%   Offset tuning satisfies both: stay on the grid, sit outside the allocation.
%
%   See also LINK_BUDGET, MEASUREMENT_SETUP.

    p = parse_options(varargin, struct( ...
        'centre_hz',      446593750, ...
        'sample_rate_hz', 2000000, ...
        'num_channels',   160, ...
        'verbose',        true));

    CHANNEL_1_HZ = 446006250;
    SPACING_HZ   = 12500;
    N_CHANNELS   = 16;

    spacing_out = p.sample_rate_hz / p.num_channels;
    steps       = (p.centre_hz - CHANNEL_1_HZ) / SPACING_HZ;
    aligned     = abs(steps - round(steps)) < 1e-9;
    on_channel  = aligned && round(steps) >= 0 && round(steps) < N_CHANNELS;

    channels   = (1:N_CHANNELS)';
    freqs      = CHANNEL_1_HZ + (channels - 1) * SPACING_HZ;
    bins       = mod(round((freqs - p.centre_hz) / spacing_out), p.num_channels);
    images     = 2 * p.centre_hz - freqs;
    image_ch   = arrayfun(@(f) channel_at(f, CHANNEL_1_HZ, SPACING_HZ, N_CHANNELS), images);

    dc_channel = channel_at(p.centre_hz, CHANNEL_1_HZ, SPACING_HZ, N_CHANNELS);

    results = struct( ...
        'aligned',        aligned, ...
        'on_channel',     on_channel, ...
        'bin_spacing_hz', spacing_out, ...
        'bins',           bins, ...
        'unique_bins',    numel(unique(bins)) == N_CHANNELS, ...
        'dc_channel',     dc_channel, ...
        'image_channels', image_ch);

    if p.verbose
        fprintf('\n=== Channel plan at %.6f MHz, %.3f MS/s, %d bins ===\n\n', ...
                p.centre_hz/1e6, p.sample_rate_hz/1e6, p.num_channels);
        fprintf('bin spacing                 %8.1f Hz   %s\n', spacing_out, ...
                verdict(abs(spacing_out - SPACING_HZ) < 1e-6, 'matches PMR446', 'WRONG'));
        fprintf('steps from channel 1        %8.2f      %s\n', steps, ...
                verdict(aligned, 'integer, channels land on bins', 'HALF-BIN, channels smear'));
        fprintf('distinct bins               %8d      %s\n', numel(unique(bins)), ...
                verdict(results.unique_bins, 'no collisions', 'COLLISION'));

        if isnan(dc_channel)
            fprintf('DC spur lands on            %8s      outside the allocation\n', 'no channel');
        else
            fprintf('DC spur lands on            %8s      PERMANENT PHANTOM EMITTER\n', ...
                    sprintf('PMR%d', dc_channel));
        end

        bad_images = sum(~isnan(image_ch));
        fprintf('IQ images onto channels     %8d      %s\n', bad_images, ...
                verdict(bad_images == 0, 'all outside the allocation', 'PHANTOMS ON REAL CHANNELS'));

        fprintf('\n%-6s %14s %6s %16s %8s\n', 'chan', 'frequency MHz', 'bin', 'image MHz', 'image');
        for k = 1:N_CHANNELS
            if isnan(image_ch(k))
                img = '   -';
            else
                img = sprintf('PMR%d', image_ch(k));
            end
            fprintf('PMR%-3d %14.6f %6d %16.6f %8s\n', ...
                    channels(k), freqs(k)/1e6, bins(k), images(k)/1e6, img);
        end
    end
end


function ch = channel_at(freq_hz, channel_1_hz, spacing_hz, n_channels)
% Nearest PMR446 channel within 2 kHz, or NaN when the frequency is off-grid.
    ch = NaN;
    for k = 1:n_channels
        if abs(freq_hz - (channel_1_hz + (k-1) * spacing_hz)) <= 2000
            ch = k;
            return;
        end
    end
end


function s = verdict(ok, yes, no)
    if ok
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
            error('channel_plan:unknownOption', 'unknown option "%s"', name);
        end
        opts.(name) = args{k+1};
    end
end
