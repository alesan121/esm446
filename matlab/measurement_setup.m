function results = measurement_setup(varargin)
% MEASUREMENT_SETUP  How close a transmitter may be before the receiver stops being linear.
%
%   results = MEASUREMENT_SETUP() reports the safe separation for the default handset and
%   receiver, and what cross-polarisation buys.
%
%   results = MEASUREMENT_SETUP('separation_m', 3.0, 'cross_polarised', true, ...)
%
%   Parameters:
%     eirp_dbm          Transmitter radiated power. A handset at minimum power with its
%                       rubber antenna is about 30 dBm.
%     separation_m      Distance between the two antennas.
%     cross_polarised   Whether the antennas are at right angles.
%     cross_pol_db      Isolation from cross-polarisation. 20 dB is the conservative figure
%                       for real antennas; theory says infinite and practice never does.
%     linear_limit_dbm  Input level above which the receiver stops being linear.
%     damage_dbm        Input level above which the receiver is damaged.
%     margin_db         Headroom required below the linear limit.
%
%   The problem this exists to solve
%   --------------------------------
%   The useful test is a real transmission with known parameters, and the available space is
%   a room. At 446 MHz the free-space loss over 3 m is only 35 dB, so a handset at minimum
%   power arrives at the receiver at -5 dBm, which is exactly the HackRF's linear limit.
%   Nothing is damaged -- that threshold is 15 dB higher -- but the front end compresses and
%   every power measurement taken through it is fiction.
%
%   Distance is not the only lever, and indoors it is the one you do not have. Turning the
%   antennas at right angles to each other gives 20 dB for free, which is worth more than
%   tripling the range.
%
%   See also LINK_BUDGET, CHANNEL_PLAN.

    p = parse_options(varargin, struct( ...
        'eirp_dbm',         30.0, ...
        'separation_m',     3.0, ...
        'cross_polarised',  true, ...
        'cross_pol_db',     20.0, ...
        'linear_limit_dbm', -5.0, ...
        'damage_dbm',       10.0, ...
        'margin_db',        10.0, ...
        'verbose',          true));

    lambda_m = 299792458 / 446.09375e6;
    fspl_db  = @(d) 20 * log10(4 * pi * d / lambda_m);

    isolation_db = 0.0;
    if p.cross_polarised
        isolation_db = p.cross_pol_db;
    end

    received_dbm = p.eirp_dbm - fspl_db(p.separation_m) - isolation_db;

    % Closest approach that still leaves the required headroom, with and without turning the
    % antennas. Inverting free-space loss: d = 10^(L/20) * lambda / (4*pi).
    allowed_db  = p.linear_limit_dbm - p.margin_db;
    loss_needed = p.eirp_dbm - allowed_db - isolation_db;
    min_distance_m = 10 ^ (loss_needed / 20) * lambda_m / (4 * pi);

    loss_aligned = p.eirp_dbm - allowed_db;
    min_aligned_m = 10 ^ (loss_aligned / 20) * lambda_m / (4 * pi);

    % Free-space loss assumes both antennas are in each other's far field. Within a couple
    % of wavelengths the formula does not describe anything real: the fields are reactive,
    % coupling depends on geometry rather than distance, and cross-polarisation isolation
    % degrades badly. At 446 MHz a wavelength is 0.67 m, so anything under about 2 m is
    % outside the model regardless of what the arithmetic returns.
    far_field_m       = 3 * lambda_m;
    min_distance_m    = max(min_distance_m, far_field_m);
    min_aligned_m     = max(min_aligned_m, far_field_m);
    in_far_field      = p.separation_m >= far_field_m;

    results = struct( ...
        'received_dbm',      received_dbm, ...
        'far_field_m',       far_field_m, ...
        'in_far_field',      in_far_field, ...
        'path_loss_db',      fspl_db(p.separation_m), ...
        'isolation_db',      isolation_db, ...
        'linear',            received_dbm <= p.linear_limit_dbm, ...
        'safe_from_damage',  received_dbm <= p.damage_dbm, ...
        'min_distance_m',    min_distance_m, ...
        'min_aligned_m',     min_aligned_m);

    if p.verbose
        fprintf('\n=== Near-field measurement setup ===\n\n');
        fprintf('transmitter EIRP            %+8.1f dBm\n', p.eirp_dbm);
        fprintf('separation                  %8.1f m\n',    p.separation_m);
        fprintf('free-space loss             %8.1f dB\n',   results.path_loss_db);
        if p.cross_polarised
            fprintf('cross-polarisation         -%8.1f dB\n', isolation_db);
        else
            fprintf('cross-polarisation          %8s\n', 'none');
        end
        fprintf('received                    %+8.1f dBm\n', received_dbm);
        fprintf('\n');
        fprintf('linear limit                %+8.1f dBm   %s\n', p.linear_limit_dbm, ...
                verdict(results.linear, 'within', 'COMPRESSING'));
        fprintf('damage threshold            %+8.1f dBm   %s\n', p.damage_dbm, ...
                verdict(results.safe_from_damage, 'safe', 'DAMAGE RISK'));
        fprintf('\n');
        fprintf('far-field limit             %8.1f m   (3 wavelengths)\n', far_field_m);
        fprintf('closest usable, aligned     %8.1f m\n', min_aligned_m);
        fprintf('closest usable, crossed     %8.1f m\n', min_distance_m);
        if ~in_far_field
            fprintf('\nWARNING: %.1f m is inside the far-field limit. The free-space figures\n', ...
                    p.separation_m);
            fprintf('above do not describe this geometry and the real coupling could be\n');
            fprintf('either higher or lower. Treat them as an order of magnitude only.\n');
        end
        if ~results.linear
            fprintf('\nAt this separation the power figures are not usable. Detection,\n');
            fprintf('channel assignment, CTCSS identification and timing still are: FM\n');
            fprintf('carries nothing in amplitude and the discriminator limits before\n');
            fprintf('demodulating, so those survive compression. Absolute power does not.\n');
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
            error('measurement_setup:unknownOption', 'unknown option "%s"', name);
        end
        opts.(name) = args{k+1};
    end
end
