function [sharpened_spectrogram, freq_bins, time_axis, raw_spectrogram, inst_freq] = sst_1d(signal_1d, fs, n_fft, hop_length)
% SST_1D Computes standard 1D Synchrosqueezing Transform (SST).
%
% Syntax:
%   [sharpened_spec, freq_bins, time_axis] = sst_1d(signal_1d, fs)
%   [sharpened_spec, freq_bins, time_axis] = sst_1d(signal_1d, fs, n_fft, hop_length)
%   [sharpened_spec, freq_bins, time_axis, raw_spec, inst_freq] = sst_1d(...)
%
% Description:
%   The Synchrosqueezing Transform (SST) is a specialized time-frequency reassignment
%   technique. Unlike conventional reassignment methods, SST only reassigns energy
%   along the frequency axis, which preserves the temporal causality and phase
%   structure of the signal while dramatically sharpening blurry time-frequency ridges.
%
%   How it works:
%   1. Computes the Short-Time Fourier Transform (STFT) with a standard Hamming
%      window w(t) to get Zxx(t, omega).
%   2. Computes the STFT with the derivative window dw(t) = d(w)/dt to get Zxx_dw(t, omega).
%   3. Evaluates the local instantaneous frequency for every (time, freq) bin via:
%          omega_inst(t, omega) = omega - (1 / 2*pi) * Im( Zxx_dw(t, omega) / Zxx(t, omega) )
%   4. Reallocates the spectral energy abs(Zxx(t, omega)) to the discrete frequency
%      bin closest to omega_inst, compressing blurred spectral lobes into razor-thin ridges.
%
% Inputs:
%   signal_1d  - 1D input signal vector (real or complex, shape N x 1).
%   fs         - Sampling frequency in Hz (e.g. 1000.0 Hz for UoG radar). Must be > 0.
%   n_fft      - (Optional) FFT window length. Default: 256.
%   hop_length - (Optional) Hop size between frames. Default: 64.
%
% Outputs:
%   sharpened_spectrogram - 2D matrix of shape (n_fft x n_frames) containing reassigned energy.
%   freq_bins             - 1D vector of frequency bin centers in Hz (length n_fft).
%   time_axis             - 1D vector of frame center times in seconds (length n_frames).
%   raw_spectrogram       - (Optional) Conventional standard STFT magnitude matrix |Zxx|.
%   inst_freq             - (Optional) Estimated instantaneous frequency matrix (Hz).
%
% Author: FracSST-Mamba Team
% Reference: Daubechies, Lu, Wu, "Synchrosqueezed Wavelet Transforms: an Empirical
%            Mode Decomposition-like Tool", Applied and Computational Harmonic Analysis, 2011.

    % Input validation
    signal_1d = signal_1d(:);
    N = length(signal_1d);
    if N == 0
        error('sst_1d: Input signal is empty.');
    end
    if nargin < 2 || fs <= 0
        error('sst_1d: Sampling frequency fs must be a positive number.');
    end
    if nargin < 3 || isempty(n_fft)
        n_fft = 256;
    end
    if nargin < 4 || isempty(hop_length)
        hop_length = 64;
    end

    if n_fft <= 0
        error('sst_1d: n_fft must be positive.');
    end
    if hop_length <= 0
        error('sst_1d: hop_length must be positive.');
    end

    % Ensure complex double precision
    sig_complex = complex(double(signal_1d));

    % 1. Prepare analysis window and its continuous time derivative
    % Hamming window w(t)
    w = hamming(n_fft);
    % Analytic time derivative dw/dt computed via centered differences with step 1/fs
    dw = gradient(w, 1.0 / fs);

    % 2. Symmetric zero-padding for border boundary conditions
    pad_len = floor(n_fft / 2);
    f_pad = [zeros(pad_len, 1); sig_complex; zeros(pad_len, 1)];

    % 3. Extract overlapping sliding window frames
    start_indices = 1:hop_length:(length(f_pad) - n_fft + 1);
    n_frames = length(start_indices);

    % Pre-allocate STFT matrices: shape (n_fft, n_frames)
    Zxx    = zeros(n_fft, n_frames);
    Zxx_dw = zeros(n_fft, n_frames);

    % Compute STFT with standard window and derivative window
    for j = 1:n_frames
        idx = start_indices(j);
        frame = f_pad(idx:(idx + n_fft - 1));

        % FFT shifted so 0 Hz is centered
        Zxx(:, j)    = fftshift(fft(frame .* w));
        Zxx_dw(:, j) = fftshift(fft(frame .* dw));
    end

    % Standard two-sided centered frequency bin grid in Hz [-fs/2, fs/2)
    k = (-floor(n_fft / 2) : ceil(n_fft / 2) - 1)';
    freq_bins = k * (fs / n_fft);
    n_freq_bins = length(freq_bins);

    % 4. Estimate Instantaneous Frequency via Phase Derivative
    % Phase derivative formula: omega_inst = omega - (1/(2*pi)) * Im( Zxx_dw / Zxx )
    abs_Zxx = abs(Zxx);
    max_amp = max(abs_Zxx(:));
    if max_amp > 0
        thresh = 1e-6 * max_amp;
    else
        thresh = 1e-12;
    end

    valid_mask = (abs_Zxx >= thresh);
    ratio = zeros(n_fft, n_frames);
    ratio(valid_mask) = Zxx_dw(valid_mask) ./ (Zxx(valid_mask) + 1e-12);

    % Frequency correction offset in Hz
    freq_offset = imag(ratio) / (2.0 * pi);

    % Instantaneous frequency candidate per (bin, frame)
    freq_bins_matrix = repmat(freq_bins, 1, n_frames);
    inst_freq = freq_bins_matrix - freq_offset;

    % For below-threshold noise bins, fall back to nominal bin frequency
    inst_freq(~valid_mask) = freq_bins_matrix(~valid_mask);

    % 5. Frequency Reassignment Step
    % Reallocate energy abs_Zxx(k, j) to target frequency bin nearest to inst_freq(k, j)
    df = fs / n_fft;
    f_min = freq_bins(1);
    sharpened_spectrogram = zeros(n_freq_bins, n_frames);

    for j = 1:n_frames
        % Compute destination bin index (1-based MATLAB indexing)
        target_bins = round((inst_freq(:, j) - f_min) / df) + 1;

        % Clip indices to valid range [1, n_freq_bins]
        target_bins = min(max(target_bins, 1), n_freq_bins);

        % Accumulate energy into target ridges
        for k_idx = 1:n_freq_bins
            target_bin = target_bins(k_idx);
            sharpened_spectrogram(target_bin, j) = ...
                sharpened_spectrogram(target_bin, j) + abs_Zxx(k_idx, j);
        end
    end

    % 6. Construct time axis in seconds
    time_axis = ((0:(n_frames - 1)) * hop_length) / fs;

    % Optional raw STFT spectrogram for visual comparison
    raw_spectrogram = abs_Zxx;
end
