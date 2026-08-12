function results = cfar_geometry(varargin)
% CFAR_GEOMETRY  What a CFAR reference window costs when the spectrum is narrow.
%
%   results = CFAR_GEOMETRY() reports the shipping geometry against the number of bins.
%   results = CFAR_GEOMETRY('num_bins', 40) checks one case.
%
%   Parameters:
%     num_reference  Reference cells, split either side of the cell under test.
%     num_guard      Guard cells either side, excluded from the estimate.
%     num_bins       Bins in the spectrum the detector sees.
%     pfa            Design probability of false alarm per cell per frame.
%     os_rank_frac   Order statistic used, as a fraction of the reference cells.
%
%   The failure this quantifies
%   ---------------------------
%   CFAR estimates the noise floor from cells around the one under test, and the whole method
%   rests on those cells containing noise and not signal. That holds when the window is a
%   small part of the spectrum. It stops holding when the window is most of it.
%
%   The window here is num_reference + 2*num_guard + 1 = 29 cells. Against the receiver's 160
%   bins that is 18 % of the spectrum and unremarkable. Against a 40-bin extract it is 72 %,
%   and a strong emitter then sits inside its own reference cells: the noise estimate rises
%   with the signal, the threshold rises above it, and the detector reports nothing at all.
%
%   Observed exactly that way. A recording in which the node found six transmissions at 40 dB
%   SNR over 160 bins produced zero detections once decimated to 40 bins, with no other change.
%
%   The rule of thumb this gives: keep the window under about a quarter of the spectrum. Below
%   that the reference cells are dominated by cells the signal does not reach; above it, the
%   detector is measuring the signal and calling it noise.
%
%   See also LINK_BUDGET, CHANNEL_PLAN.

    p = parse_options(varargin, struct( ...
        'num_reference', 24, ...
        'num_guard',     2, ...
        'num_bins',      [], ...
        'pfa',           1e-8, ...
        'os_rank_frac',  0.75, ...
        'verbose',       true));

    window = p.num_reference + 2 * p.num_guard + 1;

    if isempty(p.num_bins)
        bin_counts = [1024, 512, 256, 160, 128, 80, 64, 40, 32];
    else
        bin_counts = p.num_bins;
    end

    fraction = window ./ bin_counts;
    usable   = fraction <= 0.25;

    rank  = max(1, min(p.num_reference, round(p.os_rank_frac * p.num_reference)));
    alpha = os_threshold_factor(p.num_reference, rank, p.pfa);

    % Largest reference count that keeps the window inside the quarter-spectrum rule, for
    % each bin count. Reference cells must stay even, since they are split either side.
    max_reference = 2 * floor((0.25 * bin_counts - 2 * p.num_guard - 1) / 2);
    max_reference = max(max_reference, 2);

    results = struct( ...
        'window',          window, ...
        'bin_counts',      bin_counts, ...
        'fraction',        fraction, ...
        'usable',          usable, ...
        'threshold_db',    10 * log10(alpha), ...
        'max_reference',   max_reference);

    if p.verbose
        fprintf('\n=== CFAR window against spectrum width ===\n\n');
        fprintf('reference cells   %d\n', p.num_reference);
        fprintf('guard cells       %d either side\n', p.num_guard);
        fprintf('window            %d cells\n', window);
        fprintf('threshold factor  %.2f  (%.2f dB) at P_fa = %.0e\n\n', ...
                alpha, 10 * log10(alpha), p.pfa);

        fprintf('%8s %10s %10s   %s\n', 'bins', 'window %', 'max refs', 'verdict');
        for k = 1:numel(bin_counts)
            if bin_counts(k) < window
                verdict = 'IMPOSSIBLE -- window wider than the spectrum';
            elseif usable(k)
                verdict = 'ok';
            else
                verdict = 'DEGRADED -- signal contaminates its own reference cells';
            end
            fprintf('%8d %9.0f%% %10d   %s\n', ...
                    bin_counts(k), 100 * fraction(k), max_reference(k), verdict);
        end
        fprintf('\nRule: keep the window under a quarter of the spectrum. The "max refs"\n');
        fprintf('column is the largest even reference count that satisfies it.\n');
    end
end


function alpha = os_threshold_factor(n, k, pfa)
% OS-CFAR threshold factor by bisection.
%
% For n i.i.d. exponential reference cells and the k-th order statistic,
%   P_fa(alpha) = prod_{i=0}^{k-1} (n - i) / (n - i + alpha)
% which has no elementary inverse. Bisecting in log(alpha) keeps the bracket conditioned
% across the orders of magnitude alpha spans as P_fa tightens.
    f = @(la) sum(log((n - (0:k-1)) ./ (n - (0:k-1) + exp(la)))) - log(pfa);
    lo = -20; hi = 30;
    for iter = 1:200
        mid = 0.5 * (lo + hi);
        if f(mid) > 0
            lo = mid;
        else
            hi = mid;
        end
    end
    alpha = exp(0.5 * (lo + hi));
end


function opts = parse_options(args, defaults)
    opts = defaults;
    for k = 1:2:numel(args)
        name = args{k};
        if ~isfield(opts, name)
            error('cfar_geometry:unknownOption', 'unknown option "%s"', name);
        end
        opts.(name) = args{k+1};
    end
end
