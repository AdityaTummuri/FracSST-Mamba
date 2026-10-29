% TEST_DSP Automated Unit Test Suite for FracSST-Mamba MATLAB DSP Components.
%
% Runs comprehensive numerical and mathematical validation of:
%   - frft.m (Fractional Fourier Transform)
%   - find_optimal_alpha.m (Rényi Entropy parameter search)
%   - sst_1d.m (1D Synchrosqueezing Transform)
%   - fracsst_1d.m (Fractional Synchrosqueezing Transform pipeline)
%
% Usage:
%   >> test_dsp

clc;
clear;

fprintf('====================================================================\n');
fprintf('        FracSST-Mamba: MATLAB DSP Automated Unit Test Suite         \n');
fprintf('====================================================================\n\n');

total_tests = 0;
passed_tests = 0;

fs = 1000.0;
duration = 0.5;
test_signal = create_synthetic_radar_sample('Walk', fs, duration);
N = length(test_signal);

%% --- Group 1: FrFT Mathematical Invariance Tests ---
fprintf('>> [Group 1] Testing Fractional Fourier Transform (frft.m)...\n');

% Test 1: Identity Order a = 0
total_tests = total_tests + 1;
out_0 = frft(test_signal, 0.0);
err_0 = max(abs(out_0 - test_signal));
if err_0 < 1e-6
    fprintf('   [PASS] Test 1: FrFT identity order (a = 0) returns identical signal (err = %.2e)\n', err_0);
    passed_tests = passed_tests + 1;
else
    fprintf('   [FAIL] Test 1: FrFT identity failed (err = %.2e)\n', err_0);
end

% Test 2: Fourier Limit a = 1
total_tests = total_tests + 1;
out_1 = frft(test_signal, 1.0);
expected_fft = fftshift(fft(ifftshift(test_signal))) / sqrt(N);
err_1 = max(abs(out_1 - expected_fft));
if err_1 < 1e-5
    fprintf('   [PASS] Test 2: FrFT Fourier limit (a = 1) matches scaled FFT (err = %.2e)\n', err_1);
    passed_tests = passed_tests + 1;
else
    fprintf('   [FAIL] Test 2: FrFT a=1 failed (err = %.2e)\n', err_1);
end

% Test 3: Inverse Fourier Limit a = -1
total_tests = total_tests + 1;
out_m1 = frft(test_signal, -1.0);
expected_ifft = fftshift(ifft(ifftshift(test_signal))) * sqrt(N);
err_m1 = max(abs(out_m1 - expected_ifft));
if err_m1 < 1e-5
    fprintf('   [PASS] Test 3: FrFT inverse Fourier (a = -1) matches scaled IFFT (err = %.2e)\n', err_m1);
    passed_tests = passed_tests + 1;
else
    fprintf('   [FAIL] Test 3: FrFT a=-1 failed (err = %.2e)\n', err_m1);
end

% Test 4: Parseval Energy Conservation across Arbitrary Orders
total_tests = total_tests + 1;
orig_norm = norm(test_signal);
test_orders = [0.25, 0.5, 0.75, 1.2, -0.3, -1.5];
parseval_passed = true;
max_norm_err = 0;

for ord = test_orders
    out_ord = frft(test_signal, ord);
    norm_err = abs(norm(out_ord) - orig_norm);
    if norm_err > max_norm_err
        max_norm_err = norm_err;
    end
    if norm_err > 1e-4
        parseval_passed = false;
    end
end

if parseval_passed
    fprintf('   [PASS] Test 4: Parseval L2 energy preserved across all arbitrary orders (max err = %.2e)\n', max_norm_err);
    passed_tests = passed_tests + 1;
else
    fprintf('   [FAIL] Test 4: Parseval energy conservation violated (max err = %.2e)\n', max_norm_err);
end

% Test 5: Deterministic Numerical Stability
total_tests = total_tests + 1;
run1 = frft(test_signal, 0.65);
run2 = frft(test_signal, 0.65);
if isequal(run1, run2)
    fprintf('   [PASS] Test 5: Deterministic numerical reproducibility verified\n');
    passed_tests = passed_tests + 1;
else
    fprintf('   [FAIL] Test 5: Non-deterministic output detected across runs\n');
end

%% --- Group 2: SST Synchrosqueezing Verification Tests ---
fprintf('\n>> [Group 2] Testing Synchrosqueezing Transform (sst_1d.m)...\n');

n_fft = 128;
hop_length = 32;
[sst_spec, freq_bins, time_axis] = sst_1d(test_signal, fs, n_fft, hop_length);

% Test 6: Matrix Dimensions and Axis Consistency
total_tests = total_tests + 1;
dim_check = (size(sst_spec, 1) == n_fft) && ...
            (size(sst_spec, 2) == length(time_axis)) && ...
            (length(freq_bins) == n_fft);
if dim_check
    fprintf('   [PASS] Test 6: SST output matrix dimensions (%d x %d) match grid axes\n', ...
        size(sst_spec, 1), size(sst_spec, 2));
    passed_tests = passed_tests + 1;
else
    fprintf('   [FAIL] Test 6: SST dimension mismatch (size: %s)\n', mat2str(size(sst_spec)));
end

% Test 7: Non-negativity and Numerical Validity (No NaN / Inf)
total_tests = total_tests + 1;
valid_nums = ~any(isnan(sst_spec(:))) && ~any(isinf(sst_spec(:)));
non_neg = all(sst_spec(:) >= 0);
if valid_nums && non_neg
    fprintf('   [PASS] Test 7: Reassigned spectral energy is strictly non-negative and finite (no NaN/Inf)\n');
    passed_tests = passed_tests + 1;
else
    fprintf('   [FAIL] Test 7: SST contains negative, NaN, or Inf values\n');
end

%% --- Group 3: Optimal Alpha Search & FracSST Pipeline Tests ---
fprintf('\n>> [Group 3] Testing Optimal Alpha & Full FracSST Pipeline...\n');

% Test 8: Optimal Alpha Parameter Search
total_tests = total_tests + 1;
candidate_alphas = linspace(0.0, 1.0, 11);
[opt_a, min_ent] = find_optimal_alpha(test_signal, candidate_alphas, n_fft, hop_length);
if ismember(opt_a, candidate_alphas) && ~isnan(min_ent)
    fprintf('   [PASS] Test 8: Optimal alpha search returns valid parameter alpha* = %.3f (R3 = %.4f)\n', ...
        opt_a, min_ent);
    passed_tests = passed_tests + 1;
else
    fprintf('   [FAIL] Test 8: Optimal alpha search returned invalid alpha (%f)\n', opt_a);
end

% Test 9: End-to-End FracSST Pipeline
total_tests = total_tests + 1;
[frac_spec, f_bins, t_ax, used_alpha] = fracsst_1d(test_signal, fs, n_fft, hop_length, opt_a);
frac_valid = (size(frac_spec, 1) == n_fft) && ...
             (size(frac_spec, 2) == length(t_ax)) && ...
             (used_alpha == opt_a) && ...
             all(frac_spec(:) >= 0);
if frac_valid
    fprintf('   [PASS] Test 9: End-to-end FracSST pipeline completes successfully with verified integrity\n');
    passed_tests = passed_tests + 1;
else
    fprintf('   [FAIL] Test 9: End-to-end FracSST pipeline check failed\n');
end

% Test 10: Multi-Activity Verification (Walk, Sit, Fall)
total_tests = total_tests + 1;
activities = {'Walk', 'Sit', 'Fall'};
multi_ok = true;
for act = activities
    sig_act = create_synthetic_radar_sample(act{1}, fs, 0.4);
    [sp, ~, ~] = fracsst_1d(sig_act, fs, 64, 16, 0.5);
    if isempty(sp) || any(isnan(sp(:)))
        multi_ok = false;
        break;
    end
end
if multi_ok
    fprintf('   [PASS] Test 10: Multi-activity verification passed for Walk, Sit, and Fall signals\n');
    passed_tests = passed_tests + 1;
else
    fprintf('   [FAIL] Test 10: Multi-activity test failed\n');
end

%% Summary
fprintf('\n====================================================================\n');
fprintf('                       TEST RESULTS SUMMARY                         \n');
fprintf('====================================================================\n');
fprintf('   TOTAL TESTS : %d\n', total_tests);
fprintf('   PASSED      : %d\n', passed_tests);
fprintf('   FAILED      : %d\n', total_tests - passed_tests);
if passed_tests == total_tests
    fprintf('\n   >>> ALL TESTS PASSED! DSP LAYER IS FULLY OPERATIONAL IN MATLAB. <<<\n');
else
    fprintf('\n   >>> SOME TESTS FAILED. PLEASE CHECK THE DIAGNOSTIC OUTPUT ABOVE. <<<\n');
end
fprintf('====================================================================\n\n');
