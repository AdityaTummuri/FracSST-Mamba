function [F_a] = frft(f, a)
% FRFT Computes the 1D Discrete Fractional Fourier Transform of order a.
%
% Syntax:
%   F_a = frft(f, a)
%
% Description:
%   Computes the 1D Fractional Fourier Transform (FrFT) of order 'a' for a
%   given 1D signal vector 'f' using the Ozaktas fast chirp-multiplication
%   and convolution algorithm (H.M. Ozaktas et al., IEEE Trans. Signal Proc. 1996).
%
%   The Fractional Fourier Transform generalizes the classical Fourier Transform
%   by rotating the signal representation in the time-frequency Wigner distribution
%   plane by an angle phi = a * (pi / 2).
%     - a = 0  corresponds to the Identity operator (time domain).
%     - a = 1  corresponds to the Classical Fourier Transform (frequency domain).
%     - a = 2  corresponds to Time Reversal (parity operator).
%     - a = -1 corresponds to the Inverse Fourier Transform.
%     - Any fractional 'a' represents an intermediate rotated fractional domain,
%       which optimally concentrates linear frequency modulated (LFM) chirps
%       produced by human micro-Doppler motion (e.g., limb swings, falls).
%
% Inputs:
%   f - 1D input signal (real or complex vector of length N).
%   a - Fractional order parameter (any real scalar value).
%
% Outputs:
%   F_a - Fractional Fourier transformed signal of length N with preserved
%         L2 energy norm (Parseval theorem: norm(F_a) == norm(f)).
%
% Author: FracSST-Mamba Team
% Reference: Ozaktas, Orsak, Kutay, "Digital computation of the fractional
%            Fourier transform", IEEE Trans. Sig. Proc., 1996.

    % Ensure input is a column vector
    f = f(:);
    N = length(f);
    if N == 0
        F_a = f;
        return;
    end

    % Step 1: Reduce fractional order modulo 4 into the range [-2, 2)
    % Since FrFT is periodic with period 4 (rotation of 2*pi in Wigner plane)
    a_mod = mod(a, 4.0);
    if a_mod >= 2.0
        a_mod = a_mod - 4.0;
    end

    % Step 2: Handle special integer boundary cases directly
    % a = 0: Identity operation
    if abs(a_mod) < 1e-10
        F_a = f;
        return;
    end

    % a = 1: Standard unitary Discrete Fourier Transform (FFT)
    if abs(a_mod - 1.0) < 1e-10
        F_a = fftshift(fft(ifftshift(f))) / sqrt(N);
        return;
    end

    % a = -1: Standard unitary Inverse Discrete Fourier Transform (IFFT)
    if abs(a_mod - (-1.0)) < 1e-10
        F_a = fftshift(ifft(ifftshift(f))) * sqrt(N);
        return;
    end

    % a = 2 or -2: Time reversal (parity operator)
    if abs(abs(a_mod) - 2.0) < 1e-10
        F_a = flipud(f);
        return;
    end

    % Step 3: Shift order into stable interval [-0.5, 0.5]
    % If order is outside [-0.5, 0.5], use index additivity:
    % F_{a}(f) = F_{a - 1}( F_{1}(f) ) or F_{a + 1}( F_{-1}(f) )
    if a_mod > 0.5
        F_a = frft(frft(f, 1.0), a_mod - 1.0);
        return;
    end
    if a_mod < -0.5
        F_a = frft(frft(f, -1.0), a_mod + 1.0);
        return;
    end

    % Step 4: Core Ozaktas Algorithm for order in [-0.5, 0.5]
    phi = a_mod * (pi / 2.0);
    if abs(phi) < 1e-10
        F_a = f;
        return;
    end

    tan_half = tan(phi / 2.0);
    sin_phi  = sin(phi);
    csc_phi  = 1.0 / sin_phi;

    % 4a. Up-sample signal by factor of 2 in frequency domain (zero-padding FFT)
    % This eliminates aliasing from the quadratic chirp modulation.
    F_freq = fftshift(fft(ifftshift(f)));
    pad_left  = floor(N / 2);
    pad_right = ceil(N / 2);
    F_padded  = [zeros(pad_left, 1); F_freq; zeros(pad_right, 1)];
    f_2x      = fftshift(ifft(ifftshift(F_padded))) * sqrt(2);

    % 4b. Scaled coordinate grid for 2N points centered at 0
    m = (-N:(N - 1))';
    u = m / (2.0 * sqrt(N));

    % 4c. First chirp multiplication: modulate with chirp of rate -tan(phi/2)
    g = f_2x .* exp(-1i * pi * tan_half * (u.^2));

    % 4d. Chirp convolution kernel: kernel with rate csc(phi)
    m_h = (-2*N:(2*N))';
    u_h = m_h / (2.0 * sqrt(N));
    h   = exp(1i * pi * csc_phi * (u_h.^2));

    % 4e. Fast convolution via FFT: centered 'same' size (2N elements)
    conv_res = conv(g, h, 'same') / (2.0 * sqrt(N));

    % 4f. Second chirp multiplication: modulate with chirp of rate -tan(phi/2)
    g2 = conv_res .* exp(-1i * pi * tan_half * (u.^2));

    % 4g. Constant phase normalization factor
    if sin_phi >= 0
        sgn = 1.0;
    else
        sgn = -1.0;
    end
    c_phi = exp(-1i * (pi * sgn / 4.0 - phi / 2.0)) * sqrt(abs(csc_phi));
    g_out = c_phi * g2;

    % 4h. Down-sample back to original length N (take every second sample)
    res = g_out(1:2:(2*N));
    res = res(1:N);

    % Step 5: Enforce strict Parseval L2 energy conservation
    norm_orig = norm(f);
    if norm_orig > 0
        norm_res = norm(res);
        if norm_res > 0
            res = res * (norm_orig / norm_res);
        end
    end

    F_a = res;
end
