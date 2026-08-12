function results = link_budget(varargin)
% LINK_BUDGET  Receive chain noise figure, sensitivity and detection range.
%
%   results = LINK_BUDGET() computes the budget for the chain as deployed and prints a
%   report.
%
%   results = LINK_BUDGET('lna_gain_db', 20, 'lna_nf_db', 1.0, ...) overrides any of the
%   parameters listed below.
%
%   Parameters:
%     lna_gain_db      External LNA gain (dB). Set to 0 to model the chain without it.
%     lna_nf_db        External LNA noise figure (dB).
%     cable_loss_db    Loss between antenna and LNA (dB).
%     hackrf_nf_db     HackRF One noise figure (dB).
%     bandwidth_hz     Detection bandwidth. One PMR446 channel is 12.5 kHz.
%     required_snr_db  SNR needed for detection at the CFAR design point.
%     eirp_dbm         Emitter radiated power, for the range calculation.
%     path_loss_exp    Log-distance path loss exponent. 2 is free space, 3.5 urban.
%
%   The ordering of the chain is the whole point. Friis says the first stage sets the noise
%   figure and everything after it is divided by the gain ahead of it, so an LNA at the
%   antenna is worth far more than the same LNA at the receiver. The same formula says any
%   loss *ahead* of the LNA is paid in full.
%
%   Note the units trap: noise *factors* cascade additively, noise *figures* in dB do not.
%   Adding 1.2589 and 0.0531 gives a factor of 1.3120, whose dB value is 1.18, not 1.31.
%
%   This reproduces esm446/core/rfchain.py, which is the authority because it is what the
%   node runs. Two independent implementations of the same physics is a cross-check: if they
%   disagree, one is wrong.
%
%   See also MEASUREMENT_SETUP, CHANNEL_PLAN.

    p = parse_options(varargin, struct( ...
        'lna_gain_db',     20.0, ...
        'lna_nf_db',       1.0, ...
        'cable_loss_db',   0.5, ...
        'hackrf_nf_db',    8.0, ...
        'bandwidth_hz',    12500.0, ...
        'required_snr_db', 13.0, ...
        'eirp_dbm',        29.0, ...
        'path_loss_exp',   3.5, ...
        'verbose',         true));

    % Chain from the antenna inward. Gain in dB, noise figure in dB. A passive lossy element
    % has a noise figure equal to its loss, which is why cable before the LNA costs twice.
    if p.lna_gain_db > 0
        stages = { ...
            'antenna cable',    -p.cable_loss_db,  p.cable_loss_db; ...
            'external LNA',      p.lna_gain_db,    p.lna_nf_db; ...
            'LNA-to-SDR cable', -0.2,              0.2; ...
            'HackRF One',        0.0,              p.hackrf_nf_db};
    else
        stages = { ...
            'antenna cable',    -p.cable_loss_db,  p.cable_loss_db; ...
            'HackRF One',        0.0,              p.hackrf_nf_db};
    end

    [nf_db, total_gain_db] = cascade_noise_figure(stages);

    % Thermal noise at 290 K is -174 dBm/Hz.
    noise_floor_dbm = -174.0 + 10 * log10(p.bandwidth_hz) + nf_db;
    mds_dbm         = noise_floor_dbm + p.required_snr_db;

    % Log-distance path loss inverted for range at the minimum detectable signal.
    lambda_m        = 299792458 / 446.09375e6;
    fspl_1m_db      = 20 * log10(4 * pi * 1.0 / lambda_m);
    excess_db       = p.eirp_dbm - mds_dbm - fspl_1m_db;
    range_m         = 10 ^ (excess_db / (10 * p.path_loss_exp));

    results = struct( ...
        'noise_figure_db',  nf_db, ...
        'total_gain_db',    total_gain_db, ...
        'noise_floor_dbm',  noise_floor_dbm, ...
        'mds_dbm',          mds_dbm, ...
        'range_m',          range_m, ...
        'stages',           {stages});

    if p.verbose
        fprintf('\n=== Link budget, PMR446 at %.5f MHz ===\n\n', 446.09375);
        fprintf('%-20s %9s %8s\n', 'stage', 'gain dB', 'NF dB');
        for k = 1:size(stages, 1)
            fprintf('%-20s %9.1f %8.1f\n', stages{k,1}, stages{k,2}, stages{k,3});
        end
        fprintf('\n');
        fprintf('cascaded noise figure   %8.2f dB\n', nf_db);
        fprintf('noise floor in %.1f kHz %8.1f dBm\n', p.bandwidth_hz/1e3, noise_floor_dbm);
        fprintf('MDS at %.0f dB SNR       %8.1f dBm\n', p.required_snr_db, mds_dbm);
        fprintf('range at n=%.1f          %8.0f m   (EIRP %.0f dBm)\n', ...
                p.path_loss_exp, range_m, p.eirp_dbm);
    end
end


function [nf_db, total_gain_db] = cascade_noise_figure(stages)
% Friis: F = F1 + (F2-1)/G1 + (F3-1)/(G1*G2) + ...
%
% Accumulated in linear noise factor. Doing it in dB is the classic way to get a
% plausible-looking wrong answer.
    total_factor  = 0.0;
    gain_so_far   = 1.0;
    total_gain_db = 0.0;
    for k = 1:size(stages, 1)
        factor       = 10 ^ (stages{k,3} / 10);
        total_factor = total_factor + (factor - 1.0) / gain_so_far;
        gain_so_far  = gain_so_far * 10 ^ (stages{k,2} / 10);
        total_gain_db = total_gain_db + stages{k,2};
    end
    nf_db = 10 * log10(1.0 + total_factor);
end


function opts = parse_options(args, defaults)
% Minimal name/value parsing, so the scripts run identically in MATLAB and Octave.
    opts = defaults;
    for k = 1:2:numel(args)
        name = args{k};
        if ~isfield(opts, name)
            error('link_budget:unknownOption', 'unknown option "%s"', name);
        end
        opts.(name) = args{k+1};
    end
end
