function results = geolocation_monte_carlo(varargin)
% GEOLOCATION_MONTE_CARLO  Range from received power, with skew-aware uncertainty.
%
%   results = GEOLOCATION_MONTE_CARLO() estimates the range to an emitter from a received
%   power, propagating uncertainty by Monte Carlo, and prints the credible rings.
%
%   results = GEOLOCATION_MONTE_CARLO('received_dbm', -95, 'draws', 50000, ...) overrides
%   any of the parameters below.
%
%   Parameters:
%     received_dbm       Power at the receiver (dBm).
%     frequency_hz       Carrier frequency (Hz).
%     eirp_dbm           Assumed emitter radiated power. 27 dBm is the ETSI EN 300 296
%                        limit for PMR446.
%     eirp_sigma_db      Uncertainty in that assumption, covering both the emitter's real
%                        power and its antenna orientation.
%     path_loss_exp      Mean log-distance exponent. 2 is free space, 3.5 urban/mixed.
%     path_loss_exp_sigma Uncertainty in the exponent.
%     shadowing_sigma_db Log-normal shadowing about the mean path loss.
%     calibration_sigma_db Uncertainty in the receiver's power calibration.
%     rx_gain_dbi        Receive antenna gain.
%     draws              Monte Carlo draws.
%     seed               Random seed, for a repeatable answer.
%     verbose            Print the report. Set false to compute quietly.
%     with_sensitivity   Compute the per-assumption breakdown. Internal; the recursive
%                        calls that produce it set this false to terminate.
%
%   Why Monte Carlo rather than a propagated sigma. Distance depends exponentially on path
%   loss, so a symmetric uncertainty in decibels is a strongly skewed uncertainty in metres.
%   The linearisation in the v0 estimator, sigma_m = d * (sigma_db/(10*n)) * ln(10), puts the
%   lower ring too far out and the upper ring too close, and it treats only shadowing as
%   uncertain when the exponent -- which sits in the denominator of an exponent -- matters
%   more than anything else in the model.
%
%   The closed form this is checked against. With only shadowing uncertain, distance is
%   log-normal and the percentile has an exact expression:
%
%       d_p = d_median * 10^(z_p * sigma_db / (10 * n))
%
%   where z_p is the standard normal quantile. That identity is what
%   tests/test_matlab_consistency.py compares between this script and
%   esm446/core/geolocation.py, since two random draws cannot be compared directly.
%
%   What a single sensor cannot do. It measures range, not bearing. The product is an annulus
%   about the receiver. Drawing a point would claim a measurement nobody made.
%
%   See also LINK_BUDGET, MEASUREMENT_SETUP.

    p = parse_options(varargin, struct( ...
        'received_dbm',         -95.0, ...
        'frequency_hz',         446.09375e6, ...
        'eirp_dbm',             27.0, ...
        'eirp_sigma_db',        3.0, ...
        'path_loss_exp',        3.5, ...
        'path_loss_exp_sigma',  0.5, ...
        'shadowing_sigma_db',   8.0, ...
        'calibration_sigma_db', 2.0, ...
        'rx_gain_dbi',          0.0, ...
        'draws',                20000, ...
        'seed',                 1, ...
        'with_sensitivity',     1, ...
        'verbose',              true));

    c  = 299792458.0;
    d0 = 1.0;                                  % reference distance of the model (m)

    randn('seed', p.seed);

    % The exponent is truncated at free space: a value below 2 describes a waveguide, not an
    % outdoor path, and a draw down there turns the same measurement into kilometres.
    n = p.path_loss_exp + p.path_loss_exp_sigma * randn(p.draws, 1);
    n = max(n, 2.0);

    eirp        = p.eirp_dbm + p.eirp_sigma_db * randn(p.draws, 1);
    shadowing   = p.shadowing_sigma_db * randn(p.draws, 1);
    calibration = p.calibration_sigma_db * randn(p.draws, 1);

    % Path loss implied by the measurement, once this draw's calibration error is removed.
    observed_loss = eirp + p.rx_gain_dbi - (p.received_dbm + calibration);

    % Free-space anchor at d0, then invert 10*n*log10(d/d0) for d.
    reference_db  = 20 * log10(4 * pi * d0 * p.frequency_hz / c);
    distance_term = observed_loss - reference_db - shadowing;
    distances     = d0 * 10 .^ (distance_term ./ (10 .* n));

    wanted = [5 50 68 90 95];
    rings  = prctile_local(distances, wanted);

    % The closed form, for the shadowing-only case. Comparing a random draw against another
    % random draw proves nothing; comparing both against an exact expression proves both.
    d_median_analytic = d0 * 10 ^ ((p.eirp_dbm + p.rx_gain_dbi - p.received_dbm - ...
                                    reference_db) / (10 * p.path_loss_exp));
    d95_analytic = d_median_analytic * 10 ^ (1.6448536269514722 * p.shadowing_sigma_db / ...
                                             (10 * p.path_loss_exp));

    results = struct( ...
        'median_m',           rings(2), ...
        'p05_m',              rings(1), ...
        'p68_m',              rings(3), ...
        'p90_m',              rings(4), ...
        'p95_m',              rings(5), ...
        'd_median_analytic',  d_median_analytic, ...
        'd95_analytic',       d95_analytic, ...
        'spread',             sensitivity_or_empty(p));

    if p.verbose
        report(p, results);
    end
end

function s = sensitivity_or_empty(p)
% Skipped in the recursive calls that compute it, which would otherwise never terminate.
    if ~p.with_sensitivity
        s = struct();
        return;
    end
    s = spread_contributions(p);
end

function s = spread_contributions(p)
% How much of the interval each assumption is responsible for.
%
% Rerun with one uncertainty at a time and compare the 5-95 span. This is what tells you
% where a measurement would buy something: narrowing the assumption that dominates is worth
% doing, narrowing any of the others is not.
    names  = {'path_loss_exp_sigma', 'shadowing_sigma_db', 'eirp_sigma_db', ...
              'calibration_sigma_db'};
    s = struct();
    for k = 1:numel(names)
        q = p;
        for j = 1:numel(names)
            if j ~= k
                q.(names{j}) = 0.0;
            end
        end
        r = geolocation_monte_carlo( ...
            'received_dbm',         q.received_dbm, ...
            'frequency_hz',         q.frequency_hz, ...
            'eirp_dbm',             q.eirp_dbm, ...
            'eirp_sigma_db',        q.eirp_sigma_db, ...
            'path_loss_exp',        q.path_loss_exp, ...
            'path_loss_exp_sigma',  q.path_loss_exp_sigma, ...
            'shadowing_sigma_db',   q.shadowing_sigma_db, ...
            'calibration_sigma_db', q.calibration_sigma_db, ...
            'rx_gain_dbi',          q.rx_gain_dbi, ...
            'draws',                q.draws, ...
            'seed',                 q.seed, ...
            'with_sensitivity',     0, ...
            'verbose',              false);
        s.(names{k}) = r.p95_m / r.p05_m;
    end
end

function report(p, r)
    printf('\n=== Range from received power ===\n\n');
    printf('  received            %8.1f dBm at %.5f MHz\n', p.received_dbm, ...
           p.frequency_hz / 1e6);
    printf('  assumed EIRP        %8.1f dBm +/- %.1f dB\n', p.eirp_dbm, p.eirp_sigma_db);
    printf('  path loss exponent  %8.2f +/- %.2f\n', p.path_loss_exp, p.path_loss_exp_sigma);
    printf('  shadowing           %8.1f dB\n', p.shadowing_sigma_db);
    printf('  draws               %8d\n\n', p.draws);

    printf('  credible rings about the receiver\n');
    printf('    5%%   %10.0f m\n', r.p05_m);
    printf('   50%%   %10.0f m   <- estimate\n', r.median_m);
    printf('   68%%   %10.0f m\n', r.p68_m);
    printf('   90%%   %10.0f m\n', r.p90_m);
    printf('   95%%   %10.0f m\n\n', r.p95_m);

    ratio = r.p95_m / r.p05_m;
    printf('  the 5-95%% interval spans a factor of %.1f, which is why a single sigma\n', ratio);
    printf('  cannot describe it: the distribution is log-normal, not normal.\n\n');

    printf('  closed form, shadowing only\n');
    printf('    median          %10.0f m\n', r.d_median_analytic);
    printf('    95th percentile %10.0f m\n\n', r.d95_analytic);

    if isfield(r.spread, 'path_loss_exp_sigma')
        printf('  where the width comes from (5-95%% span with one uncertainty at a time)\n');
        printf('    path loss exponent  x%6.1f\n', r.spread.path_loss_exp_sigma);
        printf('    shadowing           x%6.1f\n', r.spread.shadowing_sigma_db);
        printf('    emitter EIRP        x%6.1f\n', r.spread.eirp_sigma_db);
        printf('    calibration         x%6.1f\n\n', r.spread.calibration_sigma_db);
        printf('  Narrowing the assumption that dominates is the only one worth measuring.\n\n');
    end

    printf('  GEOMETRY: this is a range, not a position. One omnidirectional antenna\n');
    printf('  measures no bearing, so the emitter lies somewhere on an annulus about the\n');
    printf('  receiver. Anything drawn as a point claims a measurement nobody made.\n\n');
end

function q = prctile_local(x, percentiles)
% Linear-interpolated percentiles, matching numpy.percentile's default method, so the two
% implementations can be compared without arguing about quantile conventions.
    s = sort(x(:));
    m = numel(s);
    q = zeros(size(percentiles));
    for k = 1:numel(percentiles)
        position = (percentiles(k) / 100) * (m - 1) + 1;
        lower    = floor(position);
        upper    = ceil(position);
        weight   = position - lower;
        q(k)     = s(lower) * (1 - weight) + s(upper) * weight;
    end
end

function opts = parse_options(args, defaults)
    opts = defaults;
    for k = 1:2:numel(args)
        name = args{k};
        if ~isfield(opts, name)
            error('geolocation_monte_carlo:unknownOption', 'unknown option "%s"', name);
        end
        opts.(name) = args{k + 1};
    end
end
