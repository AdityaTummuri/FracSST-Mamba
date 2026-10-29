function [sharpened_spectrogram, freq_bins, time_axis, optimal_alpha, f_rotated, raw_stft] = fracsst_1d(signal_1d, fs, n_fft, hop_length, alpha)
% FRACSST_1D Computes the 1D Fractional Synchrosqueezing Transform (FracSST).
%
% Syntax:
%   [sharpened_spec, freq_bins, time_axis, optimal_alpha] = fracsst_1d(signal_1d, fs)
%   [sharpened_spec, freq_bins, time_axis, optimal_alpha] = fracsst_1d(signal_1d, fs, n_fft, hop_length)
%   [sharpened_spec, freq_bins, time_axis, optimal_alpha] = fracsst_1d(signal_1d, fs, n_fft, hop_length, alpha)
%   [sharpened_spec, freq_bins, time_axis, optimal_alpha, f_rotated, raw_stft] = fracsst_1d(...)
%
% Description:
%   FracSST is the core Stage 1 DSP front-end of the FracSST-Mamba radar architecture.
%   Human body movements during activities (walking, sitting, falling) generate complex,
%   non-stationary Doppler frequency signatures with strong chirp-like modulations
%   due to accelerating limbs and torso displacement.
%
%   Conventional Short-Time Fourier Transforms (STFT) suffer from severe spectral smearing
%   governed by the Heisenberg-Gabor uncertainty principle.
%
%   FracSST overcomes this fundamental limitation in two synergistic stages:
%     1. Optimal Fractional Rotation (FrFT):
%        Searches for the fractional order alpha* that minimizes order-3 Rényi entropy.
%        Rotating the signal by angle phi* = alpha* * (pi/2) dechirps accelerating components,
%        transforming slanted time-frequency chirps into parallel ridges.
%     2. Synchrosqueezing Reassignment (SST):
%        Calculates the instantaneous phase derivatives and reassigns energy along the
%        frequency axis, squeezing broad spectral lobes into razor-sharp trajectories.
%
% Inputs:
%   signal_1d  - 1D input radar signal (real or complex vector of length N).
%   fs         - Radar sampling frequency in Hz (e.g. 1000 Hz).
%   n_fft      - (Optional) FFT window size. Default: 256.
%   hop_length - (Optional) Frame hop size. Default: 64.
%   alpha      - (Optional) Explicit fractional order alpha. If omitted or empty ([]),
%                the optimal alpha is automatically computed via Rényi entropy minimization.
%
% Outputs:
%   sharpened_spectrogram - 2D matrix (n_fft x n_frames) of sharpened time-frequency energy.
%   freq_bins             - 1D vector of frequency bin centers in Hz (length n_fft).
%   time_axis             - 1D vector of time positions in seconds (length n_frames).
%   optimal_alpha         - The fractional order alpha used (either supplied or searched).
%   f_rotated             - (Optional) Signal rotated into the optimal fractional domain.
%   raw_stft              - (Optional) Baseline STFT magnitude of the unrotated signal for comparison.
%
% Author: FracSST-Mamba Team

    signal_1d = signal_1d(:);
    N = length(signal_1d);
    if N == 0
        error('fracsst_1d: Input signal is empty.');
    end
    if nargin < 2 || fs <= 0
        error('fracsst_1d: Sampling frequency fs must be positive.');
    end
    if nargin < 3 || isempty(n_fft)
        n_fft = 256;
    end
    if nargin < 4 || isempty(hop_length)
        hop_length = 64;
    end

    % 1. Determine optimal fractional order alpha
    if nargin < 5 || isempty(alpha)
        search_range = linspace(0.0, 1.5, 31);
        optimal_alpha = find_optimal_alpha(signal_1d, search_range, n_fft, hop_length);
    else
        optimal_alpha = double(alpha);
    end

    % 2. Rotate signal into the optimal fractional domain
    f_rotated = frft(signal_1d, optimal_alpha);

    % 3. Apply Synchrosqueezing to the rotated signal
    [sharpened_spectrogram, freq_bins, time_axis] = sst_1d( ...
        f_rotated, ...
        fs, ...
        n_fft, ...
        hop_length ...
    );

    % 4. Optional baseline raw STFT on unrotated signal for direct before/after comparison
    if nargout >= 6
        [~, ~, ~, raw_stft] = sst_1d(signal_1d, fs, n_fft, hop_length);
    end
end
