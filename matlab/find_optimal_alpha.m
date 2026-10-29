function [best_alpha, min_entropy, search_range, entropies] = find_optimal_alpha(f, search_range, n_fft, hop_length)
% FIND_OPTIMAL_ALPHA Finds the optimal fractional order alpha via Rényi Entropy minimization.
%
% Syntax:
%   best_alpha = find_optimal_alpha(f)
%   best_alpha = find_optimal_alpha(f, search_range, n_fft, hop_length)
%   [best_alpha, min_entropy, search_range, entropies] = find_optimal_alpha(...)
%
% Description:
%   Human radar micro-Doppler motions (such as arm swings during walking or rapid
%   torso acceleration during a fall) produce non-stationary chirp-like frequency
%   trajectories. In the standard time-frequency plane (alpha = 0), these chirps
%   appear smeared due to the Heisenberg-Gabor uncertainty principle.
%
%   By rotating the signal into the Fractional Fourier domain with order alpha,
%   chirp components can be aligned with the coordinate axes, concentrating their
%   energy into compact, sharp spectral ridges.
%
%   This function performs an automated grid search over candidate alpha orders,
%   evaluating the order-3 Rényi Entropy:
%
%       R_3 = -0.5 * ln( sum( P.^3 ) )
%
%   where P is the normalized time-frequency energy distribution (P = |STFT|^2 / sum(|STFT|^2)).
%   A lower Rényi entropy indicates higher energy concentration and minimal spectral smearing.
%
% Inputs:
%   f            - 1D input radar signal vector (length N).
%   search_range - (Optional) Array of candidate alpha values to evaluate.
%                  Default: linspace(0.0, 1.5, 31).
%   n_fft        - (Optional) FFT window size for time-frequency analysis.
%                  Default: min(256, length(f)).
%   hop_length   - (Optional) Hop size between consecutive STFT frames.
%                  Default: max(1, floor(n_fft / 4)).
%
% Outputs:
%   best_alpha   - The optimal fractional order alpha yielding minimal Rényi entropy.
%   min_entropy  - The minimal entropy value attained at best_alpha.
%   search_range - Array of all evaluated candidate alpha orders.
%   entropies    - Vector of computed Rényi entropies corresponding to search_range.
%
% Author: FracSST-Mamba Team

    f = f(:);
    N = length(f);
    if N == 0
        error('find_optimal_alpha: Empty signal input.');
    end

    % Set default parameters matching project specifications
    if nargin < 2 || isempty(search_range)
        search_range = linspace(0.0, 1.5, 31);
    end
    if nargin < 3 || isempty(n_fft)
        n_fft = min(256, N);
    else
        n_fft = min(n_fft, N);
    end
    if nargin < 4 || isempty(hop_length)
        hop_length = max(1, floor(n_fft / 4));
    end

    % Pre-allocate entropy storage
    num_candidates = length(search_range);
    entropies = zeros(num_candidates, 1);
    min_entropy = Inf;
    best_alpha = search_range(1);

    % Analysis window: standard symmetric Hamming window
    w = hamming(n_fft);
    pad_len = floor(n_fft / 2);

    % Loop over each candidate fractional order alpha
    for i = 1:num_candidates
        alpha_val = search_range(i);

        % 1. Rotate signal into fractional domain of order alpha
        f_rot = frft(f, alpha_val);

        % 2. Symmetric zero-padding for border frames
        f_pad = [zeros(pad_len, 1); f_rot; zeros(pad_len, 1)];

        % 3. Extract overlapping sliding window frames
        start_indices = 1:hop_length:(length(f_pad) - n_fft + 1);
        num_frames = length(start_indices);

        % Compute STFT spectrogram magnitude squared
        spec_mag_sq = zeros(n_fft, num_frames);
        for frame_idx = 1:num_frames
            idx = start_indices(frame_idx);
            frame = f_pad(idx:(idx + n_fft - 1));
            frame_win = frame .* w;
            spec_mag_sq(:, frame_idx) = abs(fft(frame_win)).^2;
        end

        % 4. Normalize to form a 2D probability distribution P
        total_energy = sum(spec_mag_sq(:));
        if total_energy < 1e-12
            entropies(i) = Inf;
            continue;
        end

        P = spec_mag_sq / total_energy;

        % 5. Compute Rényi Entropy of order 3:
        %    R_3 = -0.5 * ln( sum( P.^3 ) )
        % Higher concentration of energy results in larger sum(P.^3), thus lower R_3.
        p3_sum = sum(P(:).^3);
        entropy_val = -0.5 * log(p3_sum + 1e-12);
        entropies(i) = entropy_val;

        % 6. Track the minimum entropy
        if entropy_val < min_entropy
            min_entropy = entropy_val;
            best_alpha = alpha_val;
        end
    end
end
