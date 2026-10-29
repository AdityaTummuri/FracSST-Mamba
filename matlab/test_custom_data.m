% TEST_CUSTOM_DATA Template for testing user radar data with the FracSST DSP layer.
%
% Project: FracSST-Mamba
%
% This script is designed for you to easily plug in ANY custom radar data
% (from a .mat file, a .csv file, a raw .dat file, or a custom variable)
% and immediately visualize how the FracSST DSP layer sharpens your signal!
%
% How to use:
%   1. Follow Method 1, 2, or 3 below to supply your data.
%   2. Set your sampling frequency 'fs' (default: 1000 Hz).
%   3. Run the script:
%      >> test_custom_data

clc;
close all;
clear;

fprintf('====================================================================\n');
fprintf('         FracSST DSP Custom Data Evaluation & Graph Generator       \n');
fprintf('====================================================================\n\n');

%% 1. Choose Your Data Input Method (Select ONE of the options below)

% =========================================================================
% METHOD 1: Use synthetic activity signals ('Walk', 'Sit', 'Fall')
% =========================================================================
fs = 1000.0;           % Sampling rate in Hz
duration = 1.5;        % Duration in seconds
my_signal = create_synthetic_radar_sample('Fall', fs, duration);
data_description = 'Synthetic Radar Fall Signal';

% =========================================================================
% METHOD 2: Load a real University of Glasgow .dat file
% (Uncomment the lines below to use a real radar recording)
% =========================================================================
% dat_filepath = '../data/raw/Dataset_848/1 December 2017 Dataset/1P36A01R01.dat';
% [my_signal, fs, meta] = uog_load_dat(dat_filepath, 'peak_bin');
% data_description = sprintf('Real Radar .dat: %s (%s)', meta.filename, meta.activity_label);

% =========================================================================
% METHOD 3: Load your own custom .mat file or .csv file
% (Uncomment and replace with your actual file name and variable)
% =========================================================================
% load('my_radar_recording.mat', 'my_custom_array');
% my_signal = my_custom_array;
% fs = 1000.0;
% data_description = 'My Custom Radar Dataset';

%% 2. Run the FracSST DSP Layer
fprintf('>> Processing: %s\n', data_description);
fprintf('   Signal Length: %d samples (%.2f seconds @ %.0f Hz)\n', ...
    length(my_signal), length(my_signal)/fs, fs);

n_fft = 256;      % Frequency resolution (256 bins)
hop_length = 64;  % Time hop size (75% overlap)

fprintf('>> Executing FracSST (searching optimal alpha and reassigning energy)...\n');
tic;
[sharp_spec, freq_bins, time_axis, best_alpha, rotated_sig, raw_stft] = ...
    fracsst_1d(my_signal, fs, n_fft, hop_length);
elapsed_time = toc;
fprintf('>> Completed in %.3f seconds! Optimal alpha = %.4f\n\n', elapsed_time, best_alpha);

%% 3. Plot Publication-Quality Before/After Comparison
fprintf('>> Generating comparison graphs...\n');

figure('Name', 'FracSST Custom Data Evaluation', ...
       'Color', [0.06 0.07 0.11], ...
       'Position', [100, 100, 1200, 550]);

col_bg = [0.10 0.11 0.18];
col_text = [0.95 0.95 0.98];

to_db = @(M) 20 * log10(abs(M) / (max(abs(M(:))) + 1e-12) + 1e-6);
raw_db   = to_db(raw_stft);
sharp_db = to_db(sharp_spec);
clim_val = [-45, 0];

% Subplot 1: Conventional STFT (Before)
ax1 = subplot(1, 2, 1);
imagesc(time_axis, freq_bins, raw_db);
axis xy; colormap(ax1, 'jet'); clim(clim_val);
cb1 = colorbar('Color', col_text);
ylabel(cb1, 'Normalized Power (dB)', 'Color', col_text);
set(ax1, 'Color', col_bg, 'XColor', col_text, 'YColor', col_text, 'FontSize', 11);
xlabel('Time (seconds)', 'Color', col_text, 'FontWeight', 'bold');
ylabel('Doppler Frequency (Hz)', 'Color', col_text, 'FontWeight', 'bold');
title({'CONVENTIONAL STFT (BEFORE)', 'Smeared Micro-Doppler Spectrum'}, ...
    'Color', [1.0 0.4 0.4], 'FontSize', 12, 'FontWeight', 'bold');
ylim([-fs/4, fs/4]);

% Subplot 2: FracSST (After)
ax2 = subplot(1, 2, 2);
imagesc(time_axis, freq_bins, sharp_db);
axis xy; colormap(ax2, 'jet'); clim(clim_val);
cb2 = colorbar('Color', col_text);
ylabel(cb2, 'Normalized Power (dB)', 'Color', col_text);
set(ax2, 'Color', col_bg, 'XColor', col_text, 'YColor', col_text, 'FontSize', 11);
xlabel('Time (seconds)', 'Color', col_text, 'FontWeight', 'bold');
ylabel('Doppler Frequency (Hz)', 'Color', col_text, 'FontWeight', 'bold');
title({sprintf('FracSST SHARPENED (AFTER, \\alpha^* = %.3f)', best_alpha), ...
       'Dechirped & Synchrosqueezed Energy Ridges'}, ...
    'Color', [0.2 1.0 0.5], 'FontSize', 12, 'FontWeight', 'bold');
ylim([-fs/4, fs/4]);

sgtitle(sprintf('DSP Performance on Custom Data: %s', data_description), ...
    'Color', 'w', 'FontSize', 14, 'FontWeight', 'bold');

fprintf('>> Done! Graph has been displayed.\n');
fprintf('   To export the plot for reports or slides, run:\n');
fprintf('   >> saveas(gcf, "my_custom_dsp_result.png");\n\n');
