% ESM446_ANALYSIS  Run every analysis script and print the results.
%
%   Usage, from this directory:
%
%       octave-cli esm446_analysis.m
%
%   or, in MATLAB or an Octave session:
%
%       esm446_analysis
%
%   Each section states the question it answers. The figures are the ones quoted in the
%   repository documentation, so if this stops agreeing with them, one of the two is stale.

fprintf('\n');
fprintf('########################################################################\n');
fprintf('# ESM-446 -- RF analysis                                               #\n');
fprintf('########################################################################\n');

% ---------------------------------------------------------------------------------------
% 1. What does the external LNA actually buy?
% ---------------------------------------------------------------------------------------
fprintf('\n\n--- 1. Sensitivity, with and without the external LNA ---\n');

with_lna    = link_budget('lna_gain_db', 20.0);
without_lna = link_budget('lna_gain_db', 0.0);

improvement_db = without_lna.mds_dbm - with_lna.mds_dbm;
range_ratio    = with_lna.range_m / without_lna.range_m;

fprintf('\nThe LNA is worth %.1f dB of sensitivity, which at n=3.5 is %.0f%% more range.\n', ...
        improvement_db, 100 * (range_ratio - 1));
fprintf('Friis: the first stage sets the noise figure, so 20 dB of gain at the antenna\n');
fprintf('divides the receiver''s own 8 dB contribution by a hundred.\n');

% ---------------------------------------------------------------------------------------
% 2. How close can a handset be before the receiver stops being linear?
% ---------------------------------------------------------------------------------------
fprintf('\n\n--- 2. Near-field test setup: 3 m indoors, handset at minimum power ---\n');

aligned = measurement_setup('separation_m', 3.0, 'cross_polarised', false);
crossed = measurement_setup('separation_m', 3.0, 'cross_polarised', true);

fprintf('\nAt 3 m the antennas cannot simply be moved apart, so the lever is polarisation.\n');
fprintf('Turning them at right angles changes %+.1f dBm into %+.1f dBm, which is the\n', ...
        aligned.received_dbm, crossed.received_dbm);
fprintf('difference between a compressed front end and a usable measurement.\n');

% ---------------------------------------------------------------------------------------
% 3. Why the receiver is not tuned to a channel
% ---------------------------------------------------------------------------------------
fprintf('\n\n--- 3. Channel plan: the centre frequency that was rejected ---\n');

rejected = channel_plan('centre_hz', 446093750);   % PMR446 channel 8

fprintf('\n--- 3b. Channel plan: offset-tuned, as shipped ---\n');
shipped = channel_plan('centre_hz', 446593750);

fprintf('\nBoth satisfy grid alignment. Only one keeps the receiver''s own artefacts out\n');
fprintf('of the allocation, and that is the constraint the first choice missed.\n');

% ---------------------------------------------------------------------------------------
% 4. What a range estimate from one sensor is actually worth
% ---------------------------------------------------------------------------------------
fprintf('\n\n--- 4. Range from received power, with its real uncertainty ---\n');

geo = geolocation_monte_carlo('received_dbm', -95);

fprintf('The interval spans a factor of %.0f. Most of that is the path loss exponent,\n', ...
        geo.p95_m / geo.p05_m);
fprintf('not the receiver: narrowing the calibration would buy almost nothing while the\n');
fprintf('exponent is an assumption rather than a measurement.\n');

% ---------------------------------------------------------------------------------------
% Summary
% ---------------------------------------------------------------------------------------
fprintf('\n\n=== Summary ===\n\n');
fprintf('noise figure with LNA        %6.2f dB\n',  with_lna.noise_figure_db);
fprintf('noise figure without LNA     %6.2f dB\n',  without_lna.noise_figure_db);
fprintf('MDS with LNA                 %6.1f dBm\n', with_lna.mds_dbm);
fprintf('MDS without LNA              %6.1f dBm\n', without_lna.mds_dbm);
fprintf('closest usable, aligned      %6.1f m\n',   aligned.min_aligned_m);
fprintf('closest usable, crossed      %6.1f m\n',   crossed.min_distance_m);
if isnan(rejected.dc_channel)
    fprintf('channel 8 as centre          usable\n');
else
    fprintf('channel 8 as centre          REJECTED: DC spur on PMR%d, %d images on channels\n', ...
            rejected.dc_channel, sum(~isnan(rejected.image_channels)));
end
if isnan(shipped.dc_channel)
    fprintf('offset-tuned centre          usable: spur and all images outside the band\n');
else
    fprintf('offset-tuned centre          REJECTED\n');
end
fprintf('range estimate at -95 dBm    %6.0f m, 5-95%% ring %.0f to %.0f m\n', ...
        geo.median_m, geo.p05_m, geo.p95_m);
fprintf('\n');
