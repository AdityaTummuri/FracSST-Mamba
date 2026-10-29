function [signal, t, activity_info] = create_synthetic_radar_sample(activity, fs, duration)
% CREATE_SYNTHETIC_RADAR_SAMPLE Generates benchmark micro-Doppler radar test signals.
%
% Syntax:
%   signal = create_synthetic_radar_sample(activity)
%   [signal, t, activity_info] = create_synthetic_radar_sample(activity, fs, duration)
%
% Description:
%   Generates benchmark synthetic 1D complex Continuous-Wave (CW) radar I/Q signals
%   modeling characteristic human activities for radar Human Activity Recognition (HAR):
%
%   - 'Walk': Combines forward torso locomotion with periodic sinusoidal
%             micro-Doppler modulations from swinging arms and legs (3 Hz gait rate).
%   - 'Sit' : Low-frequency transitional torso deceleration ending in near-static state.
%   - 'Fall': Rapid quadratic non-linear acceleration burst representing downward torso
%             impact followed by abrupt cessation of motion.
%
% Inputs:
%   activity - String: 'Walk' (default), 'Sit', or 'Fall'.
%   fs       - (Optional) Sampling frequency in Hz. Default: 1000.0 Hz.
%   duration - (Optional) Signal duration in seconds. Default: 1.0 s.
%
% Outputs:
%   signal        - 1D complex column vector [N x 1] containing In-Phase (I) and
%                   Quadrature (Q) radar baseband returns (signal = I + 1j*Q).
%   t             - Time vector in seconds.
%   activity_info - Struct containing metadata and physical motion parameters.
%
% Author: FracSST-Mamba Team

    if nargin < 1 || isempty(activity)
        activity = 'Walk';
    end
    if nargin < 2 || isempty(fs)
        fs = 1000.0;
    end
    if nargin < 3 || isempty(duration)
        duration = 1.0;
    end

    num_samples = round(fs * duration);
    t = (0:(num_samples - 1))' / fs;

    switch lower(activity)
        case 'walk'
            % Torso translation (30 Hz base Doppler) + limb swing micro-motion (3 Hz periodicity)
            f_base = 30.0;     % Base carrier Doppler (m/s equivalent)
            chirp_rate = 10.0; % Forward acceleration
            micro_freq = 3.0;  % Limb swing cadence (steps/sec)
            micro_amp = 5.0;   % Limb swing modulation amplitude

            phase = 2.0 * pi * (f_base * t + 0.5 * chirp_rate * (t.^2)) + ...
                    micro_amp * sin(2.0 * pi * micro_freq * t);
            desc = 'Walking gait: periodic sinusoidal limb swing modulations superimposed on forward torso motion';

        case 'sit'
            % Controlled transitional deceleration towards 0 Doppler
            f_base = 8.0;
            chirp_rate = 0.5;

            phase = 2.0 * pi * (f_base * t + 0.5 * chirp_rate * (t.^2));
            desc = 'Sitting down: smooth deceleration ending in low-frequency residual micro-motion';

        case 'fall'
            % Fast non-linear downward acceleration burst (sharp chirp rate)
            f_base = 20.0;
            chirp_rate = 120.0; % High quadratic acceleration

            phase = 2.0 * pi * (f_base * t + 0.5 * chirp_rate * (t.^2));
            desc = 'Fall incident: high-velocity acceleration burst followed by rapid motion cessation';

        otherwise
            error('create_synthetic_radar_sample: Unknown activity "%s". Choose "Walk", "Sit", or "Fall".', activity);
    end

    % Complex analytical radar return: I(t) + 1j * Q(t)
    signal = cos(phase) + 1i * sin(phase);

    % Metadata
    activity_info = struct();
    activity_info.activity = activity;
    activity_info.description = desc;
    activity_info.fs = fs;
    activity_info.duration = duration;
    activity_info.num_samples = num_samples;
end
