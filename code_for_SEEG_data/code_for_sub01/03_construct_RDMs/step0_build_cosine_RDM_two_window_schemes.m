%% Build cosine-distance neural RDMs with two temporal window schemes
% -------------------------------------------------------------------------
% For each responsive sEEG contact, construct one 52 x 52 neural RDM per
% time window from the 60-150 Hz HGA amplitude envelope.
%
% Two window schemes are generated independently and saved as TWO MAT files:
%
%   1) 200-ms window / 50-ms step, centers 50:50:800 ms  -> 16 columns
%   2) 150-ms window / 25-ms step, centers 50:25:800 ms  -> 31 columns
%
% Output format follows the supplied nchan_rdm example:
%
%   nchan_rdm{electrode, time_window}
%
%   - one ROW per electrode/contact
%   - non-responsive contacts remain EMPTY []
%   - each non-empty cell contains a 52 x 52 cosine-distance RDM
%
% Neural feature:
%   60-150 Hz Hilbert amplitude envelope from <subject>_NB_150.mat
%
% Baseline normalization:
%   -200 to -50 ms
%   Baseline mean and SD are estimated separately for each contact by pooling
%   baseline samples across all 52 conditions and all 12 trials.
%
% Condition pattern:
%   For each of the 52 words, baseline-normalized HGA is averaged across the
%   12 repetitions. Within each time window, its temporal samples form the
%   condition response vector.
%
% Distance:
%   cosine distance = 1 - cosine similarity
%
% Only TWO MAT files are saved, and each contains only the variable:
%   nchan_rdm
% -------------------------------------------------------------------------

clearvars;
close all;
clc;

%% ============================ USER SETTINGS =============================

subject = 'sub01';

% 当前代码所在文件夹：
% <repository>/code_for_SEEG_data/
% code_for_sub01\03_construct_RDMs
script_dir = fileparts(mfilename('fullpath'));

% 被试主目录：
% <repository>/code_for_SEEG_data/
% code_for_sub01
subject_dir = fileparts(script_dir);

% =============================== PATHS ==================================

% HGA数据：
% <repository>/code_for_SEEG_data/code_for_sub01/EEGLAB_sets\sub01_NB_150.mat
input_file = fullfile( ...
    subject_dir, ...
    'EEGLAB_sets', ...
    [subject '_NB_150.mat']);

% Responsive-contact result from Step1_plot_save_responsive_contacts.m:
resp_file = fullfile( ...
    subject_dir, ...
    '02_plot_response_responsive', ...
    'final_responsive_q005_100ms', ...
    [subject '_final_responsive_q005_100ms.mat']);

% RDM结果直接保存在当前 03_construct_RDMs 文件夹
save_dir = script_dir;
% Baseline
baseline_window = [-200 -50];  % ms

% Common analysis-center range
analysis_center_range = [50 800];  % ms

% Two temporal schemes
scheme(1).window_length = 200;
scheme(1).step_size     = 50;
scheme(1).centers       = 50:50:800;
scheme(1).tag           = '200ms_50ms';

scheme(2).window_length = 150;
scheme(2).step_size     = 25;
scheme(2).centers       = 50:25:800;
scheme(2).tag           = '150ms_25ms';

%% ================================ LOAD ==================================
assert(isfile(input_file), ...
    'Cannot find HGA input file:\n%s',input_file);

assert(isfile(resp_file), ...
    'Cannot find responsive-contact file:\n%s',resp_file);

load(input_file,'ALLEEG');
load(resp_file,'responsive_chans_col');

assert(exist('ALLEEG','var')==1, ...
    'Variable ALLEEG is missing from %s.',input_file);

assert(exist('responsive_chans_col','var')==1, ...
    'Variable responsive_chans_col is missing from %s.',resp_file);

num_conditions = numel(ALLEEG);

assert(num_conditions==52, ...
    'Expected 52 word conditions, but found %d.',num_conditions);

times = double(ALLEEG(1).times(:)');
num_timepoints = numel(times);
num_channels = size(ALLEEG(1).data,1);

% Every condition should have the same time axis and 12 trials.
for c = 1:num_conditions

    assert(isequal(double(ALLEEG(c).times(:)'),times), ...
        'Condition %d has a different time axis.',c);

    assert(ALLEEG(c).trials==12, ...
        'Condition %d has %d trials instead of 12.', ...
        c,ALLEEG(c).trials);

    assert(size(ALLEEG(c).data,1)==num_channels, ...
        'Condition %d has a different number of channels.',c);

    assert(size(ALLEEG(c).data,2)==num_timepoints, ...
        'Condition %d has a different number of time samples.',c);
end

% Responsive contacts.
% Remove the leading placeholder 0 if present, as well as any other invalid IDs.
contacts = double(responsive_chans_col(:)');
contacts = contacts( ...
    isfinite(contacts) & ...
    contacts==fix(contacts) & ...
    contacts>=1 & ...
    contacts<=num_channels);

contacts = unique(contacts,'stable');

assert(~isempty(contacts), ...
    'No valid responsive contacts were found.');

num_contacts = numel(contacts);

fprintf('\n============================================================\n');
fprintf('COSINE-DISTANCE RDM CONSTRUCTION\n');
fprintf('============================================================\n');
fprintf('Subject: %s\n',subject);
fprintf('Total channels: %d\n',num_channels);
fprintf('Responsive contacts: %d\n',num_contacts);
fprintf('Conditions: %d\n',num_conditions);
fprintf('Trials/condition: 12\n');
fprintf('Baseline: %d to %d ms\n', ...
    baseline_window(1),baseline_window(2));
fprintf('Scheme 1: 200 ms / 50 ms, 16 windows, centers 50-800 ms\n');
fprintf('Scheme 2: 150 ms / 25 ms, 31 windows, centers 50-800 ms\n');
fprintf('Output directory: %s\n',save_dir);
fprintf('============================================================\n');

%% =========================== BASELINE INDEX =============================
baseline_idx = ...
    (times >= baseline_window(1)) & ...
    (times <= baseline_window(2));

assert(nnz(baseline_idx)>=2, ...
    'Insufficient samples in the baseline interval.');

%% ========================================================================
% PRECOMPUTE ONE TRIAL-AVERAGED, BASELINE-NORMALIZED HGA RESPONSE
% FOR EACH RESPONSIVE CONTACT
%
% mean_response{ch} is 52 conditions x time samples.
% This is calculated once and reused by both window schemes.
% ========================================================================
mean_response = cell(num_channels,1);

for ic = 1:num_contacts

    ch = contacts(ic);

    fprintf('\nPreprocessing contact %d (%d/%d)...\n', ...
        ch,ic,num_contacts);

    % -------------------------------------------------------------
    % 1. Estimate contact-specific baseline mean and SD by pooling
    %    baseline samples across all conditions and all 12 trials.
    % -------------------------------------------------------------
    baseline_values = [];

    condition_data = cell(num_conditions,1);

    for c = 1:num_conditions

        x = double(ALLEEG(c).data(ch,:,:));
        x = squeeze(x);

        % Force orientation: trials x time.
        if isvector(x)

            x = reshape(x,1,[]);

        elseif size(x,1)==num_timepoints

            x = x';

        end

        assert(size(x,1)==12 && size(x,2)==num_timepoints, ...
            ['Unexpected data dimensions at contact %d, condition %d. ' ...
             'Expected trials x time = 12 x %d.'], ...
            ch,c,num_timepoints);

        condition_data{c} = x;

        b = x(:,baseline_idx);
        baseline_values = [baseline_values; b(:)]; %#ok<AGROW>
    end

    baseline_values = baseline_values(isfinite(baseline_values));

    assert(~isempty(baseline_values), ...
        'No finite baseline data at contact %d.',ch);

    mu0 = mean(baseline_values);
    sd0 = std(baseline_values,0);

    assert(isfinite(sd0) && sd0>eps, ...
        'Invalid baseline SD at contact %d.',ch);

    % -------------------------------------------------------------
    % 2. Baseline z-normalization and trial averaging.
    %
    %    Result:
    %       52 conditions x time samples
    % -------------------------------------------------------------
    Xmean = nan(num_conditions,num_timepoints);

    for c = 1:num_conditions

        z = (condition_data{c} - mu0) ./ sd0;

        Xmean(c,:) = mean(z,1,'omitnan');

    end

    mean_response{ch} = Xmean;
end

%% ========================================================================
% CONSTRUCT AND SAVE EACH WINDOW SCHEME SEPARATELY
% ========================================================================

for s = 1:numel(scheme)

    win_length = scheme(s).window_length;
    step_size  = scheme(s).step_size; %#ok<NASGU>
    centers    = scheme(s).centers;

    % Ensure requested centers match the common analysis range.
    centers = centers( ...
        centers >= analysis_center_range(1) & ...
        centers <= analysis_center_range(2));

    num_windows = numel(centers);

    fprintf('\n============================================================\n');
    fprintf('Building %s cosine RDMs\n',scheme(s).tag);
    fprintf('Window length: %d ms\n',win_length);
    fprintf('Window centers: %.0f to %.0f ms\n', ...
        centers(1),centers(end));
    fprintf('Number of windows: %d\n',num_windows);
    fprintf('============================================================\n');

    % -------------------------------------------------------------
    % Output cell:
    %   rows    = ALL channels
    %   columns = time windows
    %
    % Non-responsive rows remain empty.
    % -------------------------------------------------------------
    nchan_rdm = cell(num_channels,num_windows);

    for ic = 1:num_contacts

        ch = contacts(ic);

        Xmean = mean_response{ch};

        fprintf('Contact %d (%d/%d)\n', ...
            ch,ic,num_contacts);

        for w = 1:num_windows

            center_ms = centers(w);

            window_start = center_ms - win_length/2;
            window_end   = center_ms + win_length/2;

            % Half-open interval [start, end)
            tidx = ...
                (times >= window_start) & ...
                (times <  window_end);

            assert(nnz(tidx)>=2, ...
                ['Too few samples for contact %d, window %d ' ...
                 '(center %.1f ms).'], ...
                ch,w,center_ms);

            % Condition x temporal-feature matrix.
            X = Xmean(:,tidx);

            % -----------------------------------------------------
            % Cosine-distance RDM
            %
            % similarity(i,j) =
            %   Xi*Xj' / (||Xi|| ||Xj||)
            %
            % distance = 1 - similarity
            % -----------------------------------------------------
            norms = sqrt(sum(X.^2,2));

            assert(all(isfinite(norms) & norms>eps), ...
                ['At least one condition has an invalid/zero-norm ' ...
                 'response vector at contact %d, window center %.1f ms.'], ...
                ch,center_ms);

            Xn = X ./ norms;

            similarity = Xn * Xn';

            % Numerical cleanup.
            similarity = min(max(similarity,-1),1);

            RDM = 1 - similarity;

            % Enforce exact symmetry and zero diagonal.
            RDM = (RDM + RDM') ./ 2;
            RDM(1:num_conditions+1:end) = 0;

            nchan_rdm{ch,w} = RDM;
        end
    end

    %% ============================== SAVE ================================
    output_file = fullfile(save_dir, ...
        sprintf('%s_nchan_HGA_cosine_RDM_%s.mat', ...
        subject,scheme(s).tag));

    % Save ONLY nchan_rdm to reproduce the requested data structure.
    save(output_file,'nchan_rdm','-v7.3');

    fprintf('\nSaved:\n%s\n',output_file);
    fprintf('nchan_rdm size: %d x %d\n', ...
        size(nchan_rdm,1),size(nchan_rdm,2));

    % Quick structural check.
    nonempty_count = nnz(~cellfun(@isempty,nchan_rdm));

    fprintf('Non-empty cells: %d = %d contacts x %d windows\n', ...
        nonempty_count,num_contacts,num_windows);
end

%% ============================= FINISHED =================================
fprintf('\n============================================================\n');
fprintf('DONE. Exactly two MAT files were created:\n');
fprintf('1. %s_nchan_HGA_cosine_RDM_200ms_50ms.mat\n',subject);
fprintf('2. %s_nchan_HGA_cosine_RDM_150ms_25ms.mat\n',subject);
fprintf('\nEach MAT file contains only: nchan_rdm\n');
fprintf('Non-responsive electrode rows remain empty.\n');
fprintf('============================================================\n');
