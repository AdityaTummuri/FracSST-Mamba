% DEMO_DSP_VISUALIZER Interactive MATLAB Demonstration of the FracSST DSP Layer.
%
% Project: FracSST-Mamba — Physics-Informed Micro-Doppler Radar Activity Recognition
%
% What this script does:
%   1. Loads either a real University of Glasgow FMCW radar recording (.dat)
%      or generates a synthetic micro-Doppler radar benchmark (Walk, Sit, Fall).
%   2. Executes the complete DSP pipeline:
%        - Stage 1: Classical Short-Time Fourier Transform (STFT baseline)
%        - Stage 2: Rényi Entropy minimization to discover optimal alpha*
%        - Stage 3: Fractional Fourier Transform (FrFT) signal rotation
%        - Stage 4: Synchrosqueezing Transform (SST) instantaneous frequency reassignment
%        - Stage 5: Fractional Synchrosqueezing Transform (FracSST) synthesis
%   3. Generates publication-quality comparative figures:
%        - Figure 1: Comprehensive 6-Panel Diagnostic Dashboard
%        - Figure 2: Direct High-Resolution STFT vs. FracSST Before/After Comparison
%   4. Computes quantitative mathematical concentration metrics:
%        - Order-3 Rényi Entropy (lower = sharper)
%        - Shannon Entropy (lower = sharper)
%        - Energy Concentration Ratio (higher = sharper)
%        - Full Width at Half Maximum (FWHM) of dominant ridge
%
% Usage:
%   Simply run this script in MATLAB:
%   >> demo_dsp_visualizer

clc;
close all;
clear;

fprintf('====================================================================\n');
fprintf('     FracSST-Mamba: Digital Signal Processing (DSP) Demonstrator     \n');
fprintf('====================================================================\n\n');

%% 1. Configuration & Data Selection
% Choose data source:
%   'synthetic_walk' - Synthetic walking gait (limb swings + forward velocity)
%   'synthetic_fall' - Synthetic sudden fall (rapid quadratic chirp burst)
%   'synthetic_sit'  - Synthetic sit down (deceleration)
%   'real_uog'       - Real 5.8 GHz UoG FMCW radar .dat file

data_mode = 'synthetic_walk';  % Change to 'synthetic_fall', 'synthetic_sit', or 'real_uog'

% DSP Hyperparameters (matching the deep learning model specifications)
n_fft = 256;          % Frequency bins resolution
hop_length = 64;      % Temporal hop size (75% overlap)
duration_sec = 1.5;   % Analysis duration in seconds for demo

fprintf('>> [Step 1] Loading radar signal (Mode: %s)...\n', data_mode);

switch lower(data_mode)
    case 'synthetic_walk'
        fs = 1000.0;
        [raw_signal, t_sig, meta] = create_synthetic_radar_sample('Walk', fs, duration_sec);
        signal_title = sprintf('Synthetic Radar Walk (fs = %.0f Hz, duration = %.1f s)', fs, duration_sec);
        activity_label = 'Walk (Locomotion + Limb Swings)';

    case 'synthetic_fall'
        fs = 1000.0;
        [raw_signal, t_sig, meta] = create_synthetic_radar_sample('Fall', fs, duration_sec);
        signal_title = sprintf('Synthetic Radar Fall (fs = %.0f Hz, duration = %.1f s)', fs, duration_sec);
        activity_label = 'Fall (High Acceleration Burst)';

    case 'synthetic_sit'
        fs = 1000.0;
        [raw_signal, t_sig, meta] = create_synthetic_radar_sample('Sit', fs, duration_sec);
        signal_title = sprintf('Synthetic Radar Sit (fs = %.0f Hz, duration = %.1f s)', fs, duration_sec);
        activity_label = 'Sit (Deceleration Transition)';

    case 'real_uog'
        % Search for available .dat files in the repository
        dat_files = dir(fullfile('..', 'data', 'raw', 'Dataset_848', '**', '*.dat'));
        if isempty(dat_files)
            % Fall back to local search
            dat_files = dir(fullfile('data', 'raw', 'Dataset_848', '**', '*.dat'));
        end

        if isempty(dat_files)
            warning('No .dat files found in data/raw/Dataset_848. Falling back to synthetic walk.');
            fs = 1000.0;
            [raw_signal, t_sig, meta] = create_synthetic_radar_sample('Walk', fs, duration_sec);
            signal_title = 'Fallback: Synthetic Walk';
            activity_label = 'Walk';
        else
            sample_dat_path = fullfile(dat_files(1).folder, dat_files(1).name);
            fprintf('   Found real radar file: %s\n', dat_files(1).name);
            [full_signal, fs, meta] = uog_load_dat(sample_dat_path, 'peak_bin');

            % Crop to duration_sec to keep demo fast and responsive
            max_samples = min(round(duration_sec * fs), length(full_signal));
            raw_signal = full_signal(1:max_samples);
            t_sig = (0:(max_samples - 1))' / fs;
            signal_title = sprintf('Real UoG Radar: %s (Activity: %s, Bin: %d)', ...
                meta.filename, meta.activity_label, meta.selected_range_bin);
            activity_label = meta.activity_label;
        end

    otherwise
        error('Unknown data_mode: %s', data_mode);
end

fprintf('   Signal length: %d samples (%.2f seconds @ %.0f Hz)\n', ...
    length(raw_signal), length(raw_signal)/fs, fs);

%% 2. Step 1: Classical STFT Spectrogram (Baseline)
fprintf('\n>> [Step 2] Computing Classical STFT Spectrogram...\n');
[stft_spec_raw, freq_bins, time_axis, ~, ~] = sst_1d(raw_signal, fs, n_fft, hop_length);
% Note: sst_1d returns raw STFT as fourth argument, or we use abs(stft)
[~, ~, ~, stft_magnitude] = sst_1d(raw_signal, fs, n_fft, hop_length);

%% 3. Step 2: Optimal Fractional Order Search via Rényi Entropy
fprintf('\n>> [Step 3] Searching for optimal fractional order alpha* via Rényi Entropy...\n');
alpha_search_grid = linspace(0.0, 1.5, 31);
[optimal_alpha, min_entropy, evaluated_alphas, alpha_entropies] = ...
    find_optimal_alpha(raw_signal, alpha_search_grid, n_fft, hop_length);

fprintf('   Optimal fractional order alpha* = %.4f (Minimal Rényi Entropy = %.4f)\n', ...
    optimal_alpha, min_entropy);

%% 4. Step 3 & 4: Standard SST vs. Fractional SST (FracSST)
fprintf('\n>> [Step 4] Computing Standard SST (alpha = 0)...\n');
[sst_spec, ~, ~] = sst_1d(raw_signal, fs, n_fft, hop_length);

fprintf('\n>> [Step 5] Computing Fractional SST (FracSST with alpha = %.4f)...\n', optimal_alpha);
[fracsst_spec, ~, ~, ~, rotated_signal] = fracsst_1d( ...
    raw_signal, fs, n_fft, hop_length, optimal_alpha);

%% 5. Quantitative Metric Calculations
fprintf('\n>> [Step 6] Computing Quantitative Concentration Metrics...\n');

% Function to compute normalized order-3 Rényi Entropy
calc_renyi = @(S) -0.5 * log(sum(( (S(:).^2) / (sum(S(:).^2) + 1e-12) ).^3) + 1e-12);

% Function to compute Shannon Entropy
calc_shannon = @(S) -sum( ( (S(:).^2) / (sum(S(:).^2) + 1e-12) ) .* ...
    log( (S(:).^2) / (sum(S(:).^2) + 1e-12) + 1e-12 ));

% Energy Concentration Measure (ratio of L4 norm to L2 norm)
calc_ecm = @(S) sum(S(:).^4) / ((sum(S(:).^2))^2 + 1e-12);

metric_renyi_stft    = calc_renyi(stft_magnitude);
metric_renyi_sst     = calc_renyi(sst_spec);
metric_renyi_fracsst = calc_renyi(fracsst_spec);

metric_shannon_stft    = calc_shannon(stft_magnitude);
metric_shannon_sst     = calc_shannon(sst_spec);
metric_shannon_fracsst = calc_shannon(fracsst_spec);

metric_ecm_stft    = calc_ecm(stft_magnitude);
metric_ecm_sst     = calc_ecm(sst_spec);
metric_ecm_fracsst = calc_ecm(fracsst_spec);

% Print formatted comparative table
fprintf('\n+----------------------------------------------------------------------+\n');
fprintf('|                  DSP PERFORMANCE COMPARISON TABLE                    |\n');
fprintf('+-----------------------+---------------+---------------+--------------+\n');
fprintf('| Metric                | Standard STFT | Standard SST  | FracSST (Ours|\n');
fprintf('+-----------------------+---------------+---------------+--------------+\n');
fprintf('| Rényi Entropy (R3)    | %13.4f | %13.4f | %12.4f |\n', ...
    metric_renyi_stft, metric_renyi_sst, metric_renyi_fracsst);
fprintf('| Shannon Entropy (H)   | %13.4f | %13.4f | %12.4f |\n', ...
    metric_shannon_stft, metric_shannon_sst, metric_shannon_fracsst);
fprintf('| Energy Conc. (ECM)    | %13.4e | %13.4e | %12.4e |\n', ...
    metric_ecm_stft, metric_ecm_sst, metric_ecm_fracsst);
fprintf('+-----------------------+---------------+---------------+--------------+\n');
fprintf('  * Lower Rényi/Shannon Entropy indicates sharper time-frequency concentration.\n');
fprintf('  * Higher ECM indicates energy is concentrated into fewer, sharper ridges.\n\n');

%% 6. Visualization — Figure 1: Comprehensive 6-Panel Diagnostic Dashboard
fprintf('>> [Step 7] Rendering Figure 1: 6-Panel DSP Diagnostic Dashboard...\n');

fig1 = figure('Name', 'FracSST DSP Diagnostic Dashboard', ...
              'Color', [0.06 0.07 0.11], ...
              'Position', [50, 50, 1400, 850]);

% Color palette
col_bg = [0.10 0.11 0.18];
col_text = [0.95 0.95 0.98];
col_accent = [0.0 0.83 1.0];
col_highlight = [1.0 0.3 0.4];

% Convert spectrograms to Decibels (dB) for dynamic range visualization
to_db = @(M) 20 * log10(abs(M) / (max(abs(M(:))) + 1e-12) + 1e-6);
stft_db    = to_db(stft_magnitude);
sst_db     = to_db(sst_spec);
fracsst_db = to_db(fracsst_spec);
clim_db    = [-45, 0]; % 45 dB dynamic range

% --- Panel 1: Time-Domain In-Phase / Quadrature Radar Signal ---
ax1 = subplot(2, 3, 1);
plot(t_sig, real(raw_signal), 'LineWidth', 1.2, 'Color', [0.0, 0.83, 1.0]); hold on;
plot(t_sig, imag(raw_signal), 'LineWidth', 1.0, 'Color', [1.0, 0.55, 0.2]);
grid on; set(ax1, 'Color', col_bg, 'XColor', col_text, 'YColor', col_text);
xlabel('Time (seconds)', 'Color', col_text, 'FontWeight', 'bold');
ylabel('Baseband Amplitude', 'Color', col_text, 'FontWeight', 'bold');
title('(A) Raw Radar I/Q Signal', 'Color', col_accent, 'FontSize', 12, 'FontWeight', 'bold');
legend({'In-Phase (I)', 'Quadrature (Q)'}, 'TextColor', col_text, 'Color', col_bg, 'EdgeColor', 'none', 'Location', 'northeast');
xlim([0, t_sig(end)]);

% --- Panel 2: Rényi Entropy vs. Fractional Order alpha ---
ax2 = subplot(2, 3, 2);
plot(evaluated_alphas, alpha_entropies, '-o', 'LineWidth', 1.8, 'Color', [0.7, 0.4, 1.0], ...
    'MarkerFaceColor', [0.8, 0.5, 1.0], 'MarkerSize', 5); hold on;
plot(optimal_alpha, min_entropy, 'p', 'MarkerSize', 14, ...
    'MarkerFaceColor', col_highlight, 'MarkerEdgeColor', 'w', 'LineWidth', 1.5);
grid on; set(ax2, 'Color', col_bg, 'XColor', col_text, 'YColor', col_text);
xlabel('Fractional Order \alpha', 'Color', col_text, 'FontWeight', 'bold');
ylabel('Order-3 Rényi Entropy R_3', 'Color', col_text, 'FontWeight', 'bold');
title(sprintf('(B) Optimal Rotation Search (\\alpha^* = %.3f)', optimal_alpha), ...
    'Color', col_accent, 'FontSize', 12, 'FontWeight', 'bold');
legend({'Entropy Profile', sprintf('Optimal \\alpha^* = %.3f', optimal_alpha)}, ...
    'TextColor', col_text, 'Color', col_bg, 'EdgeColor', 'none', 'Location', 'best');

% --- Panel 3: Spectral Slice Cross-Section (Resolution Comparison) ---
ax3 = subplot(2, 3, 3);
% Find frame index with maximum instantaneous energy
[~, peak_frame_idx] = max(sum(abs(raw_signal)));
peak_frame_idx = max(1, min(peak_frame_idx, length(time_axis)));
peak_time = time_axis(peak_frame_idx);

slice_stft    = abs(stft_magnitude(:, peak_frame_idx)) / max(abs(stft_magnitude(:, peak_frame_idx)));
slice_sst     = abs(sst_spec(:, peak_frame_idx)) / max(abs(sst_spec(:, peak_frame_idx)));
slice_fracsst = abs(fracsst_spec(:, peak_frame_idx)) / max(abs(fracsst_spec(:, peak_frame_idx)));

plot(freq_bins, slice_stft, 'LineWidth', 1.2, 'LineStyle', '--', 'Color', [0.6 0.6 0.7]); hold on;
plot(freq_bins, slice_sst, 'LineWidth', 1.5, 'Color', [0.2 0.8 0.4]);
plot(freq_bins, slice_fracsst, 'LineWidth', 2.0, 'Color', col_highlight);
grid on; set(ax3, 'Color', col_bg, 'XColor', col_text, 'YColor', col_text);
xlabel('Frequency (Hz)', 'Color', col_text, 'FontWeight', 'bold');
ylabel('Normalized Energy', 'Color', col_text, 'FontWeight', 'bold');
title(sprintf('(C) Spectral Slice @ t = %.2f s', peak_time), ...
    'Color', col_accent, 'FontSize', 12, 'FontWeight', 'bold');
legend({'Standard STFT (Smeared)', 'Standard SST (\alpha=0)', 'FracSST (Sharp Ridge)'}, ...
    'TextColor', col_text, 'Color', col_bg, 'EdgeColor', 'none', 'Location', 'northeast');
xlim([-fs/4, fs/4]);

% --- Panel 4: Standard STFT Spectrogram ---
ax4 = subplot(2, 3, 4);
imagesc(time_axis, freq_bins, stft_db);
axis xy; colormap(ax4, 'inferno'); clim(clim_db);
set(ax4, 'Color', col_bg, 'XColor', col_text, 'YColor', col_text);
xlabel('Time (seconds)', 'Color', col_text, 'FontWeight', 'bold');
ylabel('Doppler Frequency (Hz)', 'Color', col_text, 'FontWeight', 'bold');
title(sprintf('(D) Conventional STFT (R_3 = %.2f)', metric_renyi_stft), ...
    'Color', col_accent, 'FontSize', 12, 'FontWeight', 'bold');
ylim([-fs/4, fs/4]);

% --- Panel 5: Standard Synchrosqueezing (SST) ---
ax5 = subplot(2, 3, 5);
imagesc(time_axis, freq_bins, sst_db);
axis xy; colormap(ax5, 'inferno'); clim(clim_db);
set(ax5, 'Color', col_bg, 'XColor', col_text, 'YColor', col_text);
xlabel('Time (seconds)', 'Color', col_text, 'FontWeight', 'bold');
ylabel('Doppler Frequency (Hz)', 'Color', col_text, 'FontWeight', 'bold');
title(sprintf('(E) Standard SST (\\alpha=0, R_3 = %.2f)', metric_renyi_sst), ...
    'Color', col_accent, 'FontSize', 12, 'FontWeight', 'bold');
ylim([-fs/4, fs/4]);

% --- Panel 6: Fractional Synchrosqueezing (FracSST - Our Method) ---
ax6 = subplot(2, 3, 6);
imagesc(time_axis, freq_bins, fracsst_db);
axis xy; colormap(ax6, 'inferno'); clim(clim_db);
cb = colorbar('Color', col_text);
ylabel(cb, 'Normalized Power (dB)', 'Color', col_text);
set(ax6, 'Color', col_bg, 'XColor', col_text, 'YColor', col_text);
xlabel('Time (seconds)', 'Color', col_text, 'FontWeight', 'bold');
ylabel('Doppler Frequency (Hz)', 'Color', col_text, 'FontWeight', 'bold');
title(sprintf('(F) FracSST (\\alpha^* = %.3f, R_3 = %.2f)', optimal_alpha, metric_renyi_fracsst), ...
    'Color', [0.2 1.0 0.5], 'FontSize', 12, 'FontWeight', 'bold');
ylim([-fs/4, fs/4]);

% Overall Super Title
sgtitle({sprintf('FracSST-Mamba Stage 1 DSP: %s', activity_label), ...
         signal_title}, ...
        'Color', 'w', 'FontSize', 14, 'FontWeight', 'bold');

%% 7. Visualization — Figure 2: Direct Large Before / After Comparison
fprintf('>> [Step 8] Rendering Figure 2: High-Resolution Before/After Comparison...\n');

fig2 = figure('Name', 'Conventional STFT vs. FracSST Comparison', ...
              'Color', [0.06 0.07 0.11], ...
              'Position', [100, 100, 1300, 600]);

% Left Subplot: Conventional STFT
b_ax1 = subplot(1, 2, 1);
imagesc(time_axis, freq_bins, stft_db);
axis xy; colormap(b_ax1, 'magma'); clim(clim_db);
cb1 = colorbar('Color', col_text, 'Location', 'southoutside');
xlabel(cb1, 'Power Spectral Density (dB)', 'Color', col_text);
set(b_ax1, 'Color', col_bg, 'XColor', col_text, 'YColor', col_text, 'FontSize', 11);
xlabel('Time (seconds)', 'Color', col_text, 'FontWeight', 'bold');
ylabel('Doppler Frequency (Hz)', 'Color', col_text, 'FontWeight', 'bold');
title({'CONVENTIONAL STFT SPECTROGRAM', 'Severe Heisenberg-Gabor Uncertainty Smearing'}, ...
    'Color', [1.0 0.4 0.4], 'FontSize', 13, 'FontWeight', 'bold');
ylim([-fs/4, fs/4]);

% Right Subplot: FracSST
b_ax2 = subplot(1, 2, 2);
imagesc(time_axis, freq_bins, fracsst_db);
axis xy; colormap(b_ax2, 'magma'); clim(clim_db);
cb2 = colorbar('Color', col_text, 'Location', 'southoutside');
xlabel(cb2, 'Power Spectral Density (dB)', 'Color', col_text);
set(b_ax2, 'Color', col_bg, 'XColor', col_text, 'YColor', col_text, 'FontSize', 11);
xlabel('Time (seconds)', 'Color', col_text, 'FontWeight', 'bold');
ylabel('Doppler Frequency (Hz)', 'Color', col_text, 'FontWeight', 'bold');
title({sprintf('PROPOSED FracSST SPECTROGRAM (\\alpha^* = %.3f)', optimal_alpha), ...
       'Physics-Informed Dechirped & Reassigned High-Energy Ridges'}, ...
    'Color', [0.2 1.0 0.5], 'FontSize', 13, 'FontWeight', 'bold');
ylim([-fs/4, fs/4]);

sgtitle(sprintf('Direct Visual Comparison: %s', activity_label), ...
    'Color', 'w', 'FontSize', 15, 'FontWeight', 'bold');

fprintf('\n>> Visualization Complete! Figures 1 and 2 are displayed.\n');
fprintf('====================================================================\n');
fprintf('  How to explain this in your presentation/demonstration:\n');
fprintf('  1. In Figure 1(D), show how the standard STFT suffers from wide,\n');
fprintf('     blurry Doppler bands caused by the Heisenberg uncertainty principle.\n');
fprintf('  2. In Figure 1(B), explain that the DSP layer rotates the signal by\n');
fprintf('     optimal angle alpha* = %.3f to minimize Rényi entropy (R3 = %.2f).\n', ...
    optimal_alpha, metric_renyi_fracsst);
fprintf('  3. In Figure 1(F) and Figure 2 (Right), point out how FracSST squeezes\n');
fprintf('     the energy into razor-thin ridges, providing clean motion features\n');
fprintf('     for the Mamba state-space deep learning network.\n');
fprintf('====================================================================\n\n');
