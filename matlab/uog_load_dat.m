function [slow_time_signal, fs, metadata] = uog_load_dat(filepath, mode)
% UOG_LOAD_DAT Loads and preprocesses raw University of Glasgow FMCW radar .dat files.
%
% Syntax:
%   [slow_time_signal, fs] = uog_load_dat(filepath)
%   [slow_time_signal, fs, metadata] = uog_load_dat(filepath, mode)
%
% Description:
%   Implements the raw radar signal extraction pipeline matching the project's
%   UoGRadarAdapter (data/adapters/uog_adapter.py):
%
%   Hardware Radar Specifications (University of Glasgow INSHEP Radar):
%     - Carrier Frequency : 5.8 GHz (C-band Continuous-Wave FMCW)
%     - Chirp Duration    : 1.0 ms  (Slow-time Pulse Repetition Frequency PRF / fs = 1000 Hz)
%     - ADC Fast-Time     : 128 samples per chirp
%     - Sweep Bandwidth   : 400 MHz
%
%   Processing Stages:
%     1. Header parsing (fc, chirp duration, fast-time sample count, bandwidth).
%     2. Complex I/Q matrix construction: (num_chirps x fast_time_samples).
%     3. Range-FFT across fast-time dimension to resolve target distance bins.
%     4. Moving Target Indication (MTI) high-pass clutter filter: subtracts
%        stationary background returns along slow-time chirps.
%     5. Range bin Doppler extraction: extracts the slow-time complex Doppler signal
%        from the peak moving target range bin (or range integration).
%
% Inputs:
%   filepath - Full or relative path to the .dat file (e.g. 'data/raw/Dataset_848/.../1P36A01R01.dat').
%   mode     - (Optional) 'peak_bin' (default: selects range bin with maximum variance)
%              or 'integrated' (sums energy across all range bins).
%
% Outputs:
%   slow_time_signal - 1D complex Doppler signal array [num_chirps x 1].
%   fs               - Slow-time sampling frequency in Hz (typically 1000.0 Hz).
%   metadata         - Struct containing radar parameters, extracted range bin, and activity.
%
% Author: FracSST-Mamba Team

    if nargin < 2 || isempty(mode)
        mode = 'peak_bin';
    end

    if ~exist(filepath, 'file')
        error('uog_load_dat: File not found at "%s".', filepath);
    end

    % 1. Open and parse 4-line metadata header
    fid = fopen(filepath, 'r');
    if fid == -1
        error('uog_load_dat: Could not open file "%s".', filepath);
    end

    line0 = fgetl(fid); % Carrier frequency in Hz (e.g. 5800000000)
    line1 = fgetl(fid); % Chirp duration in ms (e.g. 1.0)
    line2 = fgetl(fid); % Fast-time ADC samples per chirp (e.g. 128)
    line3 = fgetl(fid); % Bandwidth in Hz (e.g. 400000000)

    fc = str2double(strtrim(line0));
    chirp_dur_ms = str2double(strtrim(line1));
    nsamples_per_chirp = round(str2double(strtrim(line2)));
    bandwidth = str2double(strtrim(line3));

    % Sampling frequency of slow-time Doppler sequence is the PRF:
    % fs = 1 / (chirp_duration_in_seconds) = 1 / (1.0 ms) = 1000 Hz
    fs = 1.0 / (chirp_dur_ms * 1e-3);

    % 2. Parse complex I/Q data stream
    % Format in .dat is '<real><+or-><imag>i' per line
    C = textscan(fid, '%f%fi');
    fclose(fid);

    re_part = C{1};
    im_part = C{2};

    if isempty(re_part) || isempty(im_part)
        error('uog_load_dat: Failed to parse complex numbers from file "%s".', filepath);
    end

    complex_data = complex(re_part, im_part);
    total_samples = length(complex_data);

    num_chirps = floor(total_samples / nsamples_per_chirp);
    if num_chirps == 0
        error('uog_load_dat: Insufficient samples (%d) for %d samples/chirp.', total_samples, nsamples_per_chirp);
    end

    % 3. Reshape into 2D Radar Matrix: (num_chirps, nsamples_per_chirp)
    % Note: reshape fills column-first, so reshape into (nsamples, num_chirps).'
    radar_matrix = reshape(complex_data(1:(num_chirps * nsamples_per_chirp)), ...
                           [nsamples_per_chirp, num_chirps]).';

    % 4. Fast-time Range FFT (resolves range bins along dimension 2)
    range_profiles = fft(radar_matrix, [], 2);

    % 5. Moving Target Indication (MTI) Clutter Filter
    % Subtract static mean along the slow-time dimension (dim 1)
    % This eliminates non-moving walls, floor, and furniture reflections
    mti_profiles = range_profiles - mean(range_profiles, 1);

    % 6. Extract 1D Slow-Time Doppler Signal
    if strcmpi(mode, 'peak_bin')
        % Select range bin with maximum variance/energy of moving target
        bin_powers = mean(abs(mti_profiles).^2, 1);
        [~, peak_bin] = max(bin_powers);
        slow_time_signal = mti_profiles(:, peak_bin);
        selected_bin = peak_bin;
    else
        % Sum across all range bins
        slow_time_signal = sum(mti_profiles, 2);
        selected_bin = -1;
    end

    % 7. Identify activity from filename convention
    [~, fname, ~] = fileparts(filepath);
    prefix = fname(1);
    switch prefix
        case '1'
            activity_label = 'Walk';
        case '2'
            activity_label = 'Sit';
        case '6'
            activity_label = 'Fall';
        otherwise
            activity_label = 'Unknown';
    end

    % Construct metadata struct
    metadata = struct();
    metadata.filename = fname;
    metadata.activity_label = activity_label;
    metadata.carrier_frequency_hz = fc;
    metadata.chirp_duration_ms = chirp_dur_ms;
    metadata.fast_time_samples = nsamples_per_chirp;
    metadata.bandwidth_hz = bandwidth;
    metadata.num_chirps = num_chirps;
    metadata.duration_seconds = num_chirps / fs;
    metadata.selected_range_bin = selected_bin;
    metadata.mode = mode;
end
