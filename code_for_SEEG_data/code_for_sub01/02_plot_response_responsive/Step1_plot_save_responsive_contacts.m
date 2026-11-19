%% ========================================================================
% FINAL AUDITORY WORD-RESPONSIVE CONTACT IDENTIFICATION
%
% Final criterion:
%
%   Signal:             60-150 Hz HGA
%   Baseline:           -200 to -50 ms
%   Search window:       0 to 800 ms
%   Statistical unit:   trial
%   Statistical test:   one-sided one-sample t-test
%                       H1: HGA > baseline
%   Correction:         BH-FDR q < 0.05
%                       globally across electrode x time
%   Responsive contact: >=100 ms contiguous significant positive response
%
% Outputs:
%   1. responsive_chans_col
%   2. responsive_info numeric matrix
%   3. responsive_table
%   4. latency of first qualifying response
%   5. all corrected significance masks
%   6. all >=100-ms qualifying significance masks
%   7. response plots for all responsive contacts
%
% IMPORTANT:
% ALLEEG.data is assumed to already contain 60-150 Hz HGA
% amplitude/power time series.
%
%% ========================================================================

close all;
clear;
clc;


%% ========================================================================
%% 1. LOAD DATA
%% ========================================================================

subjName = 'sub01';

rootDir = fileparts(fileparts(mfilename('fullpath')));

dataDir = fullfile(rootDir, 'EEGLAB_sets');

saveDir_base = fullfile(rootDir, '02_plot_response_responsive');

fileName = [subjName, '_NB_150.mat'];

imgPrefix = [subjName, '_NB_Responsive_Page_'];


fprintf('\n============================================================\n');
fprintf('Loading: %s\n', fileName);
fprintf('============================================================\n');

load(fullfile(dataDir, fileName));


if ~exist('ALLEEG','var')
    error('ALLEEG was not found in the MAT file.');
end


%% ========================================================================
%% 2. OUTPUT DIRECTORY
%% ========================================================================

saveDir = fullfile( ...
    saveDir_base, ...
    'final_responsive_q005_100ms');


if ~exist(saveDir,'dir')
    mkdir(saveDir);
end


fprintf('Results will be saved to:\n%s\n',saveDir);


%% ========================================================================
%% 3. PARAMETERS
%% ========================================================================

% ------------------------------------------------------------------------
% Conditions
% -------------------------------------------------------------------------

cond_list = 1:length(ALLEEG);


% ------------------------------------------------------------------------
% Time axis
% -------------------------------------------------------------------------

t = double(ALLEEG(1).times);

dt_ms = median(diff(t));

fs_hga = 1000 / dt_ms;


fprintf('\nTime resolution = %.4f ms\n',dt_ms);
fprintf('Sampling rate   = %.2f Hz\n',fs_hga);


% ------------------------------------------------------------------------
% Channels
% -------------------------------------------------------------------------

chan_max = length(ALLEEG(1).chanlocs);


% ========================================================================
% IMPORTANT:
% This is taken from your latest diagnostic code.
% Change only this line if another subject requires another exclusion list.
% ========================================================================

chans_to_remove = [113:114];


% prevent invalid indices
chans_to_remove = chans_to_remove( ...
    chans_to_remove >= 1 & ...
    chans_to_remove <= chan_max);


chan_list = setdiff(1:chan_max,chans_to_remove);


fprintf('\nTotal channels    = %d\n',chan_max);
fprintf('Excluded channels = %d\n',numel(chans_to_remove));
fprintf('Analysed channels = %d\n',numel(chan_list));


%% ========================================================================
%% 4. FINAL RESPONSIVE PARAMETERS
%% ========================================================================

% Baseline
baseline_win = [-200 -50];

% Responsive search window
responsive_search_win = [0 800];

% BH-FDR threshold
fdr_q = 0.05;

% Minimum continuous response
min_contig_ms = 100;


% ------------------------------------------------------------------------
% Plot settings
% -------------------------------------------------------------------------

plot_xlim = [-200 1000];

plot_ylim = [-0.5 3];

y_range = plot_ylim(2);

n_plot_per_page = 30;


%% ========================================================================
%% 5. TIME INDICES
%% ========================================================================

base_indices = find( ...
    t >= baseline_win(1) & ...
    t < baseline_win(2));


search_indices = find( ...
    t >= responsive_search_win(1) & ...
    t <= responsive_search_win(2));


if isempty(base_indices)
    error('Baseline window cannot be found.');
end

if isempty(search_indices)
    error('Search window cannot be found.');
end


search_t = t(search_indices);


% Number of samples required for >=100 ms
min_contig_len = ceil(min_contig_ms / dt_ms);


fprintf('\n============================================================\n');
fprintf('FINAL RESPONSIVE CRITERION\n');
fprintf('============================================================\n');

fprintf('Baseline             = %.0f to %.0f ms\n', ...
    baseline_win(1),baseline_win(2));

fprintf('Search window        = %.0f to %.0f ms\n', ...
    responsive_search_win(1), ...
    responsive_search_win(2));

fprintf('BH-FDR q             = %.3f\n',fdr_q);

fprintf('Minimum duration     = %d ms\n',min_contig_ms);

fprintf('Minimum samples      = %d consecutive samples\n', ...
    min_contig_len);


%% ========================================================================
%% 6. PREALLOCATE
%% ========================================================================

n_chan = numel(chan_list);

n_time = numel(search_indices);


% Raw one-sided t-test P values:
% [electrode x time]
p_ttest = nan(n_chan,n_time);


% Number of trials for each channel
n_trials_all = nan(n_chan,1);


%% ========================================================================
%% 7. TIME-RESOLVED ONE-SIDED T-TEST
%% ========================================================================

fprintf('\n============================================================\n');
fprintf('TIME-RESOLVED ONE-SIDED T-TEST\n');
fprintf('============================================================\n');


for ii = 1:n_chan

    nchan = chan_list(ii);


    % ====================================================================
    % Collect ALL words / conditions
    %
    % No word identity / phonological / semantic information enters
    % responsive-contact selection.
    % ====================================================================

    all_trials_z = collect_trials_zscore( ...
        ALLEEG, ...
        cond_list, ...
        nchan, ...
        t, ...
        base_indices);


    n_trials_all(ii) = size(all_trials_z,1);


    % --------------------------------------------------------------------
    % Trial-wise baseline scalar
    %
    % Because data have already been trial-wise baseline z-scored,
    % B should be approximately zero.
    % --------------------------------------------------------------------

    B = mean( ...
        all_trials_z(:,base_indices), ...
        2, ...
        'omitnan');


    % --------------------------------------------------------------------
    % Test every time point from 0 to 800 ms
    % --------------------------------------------------------------------

    for kk = 1:n_time

        tidx = search_indices(kk);


        x_t = all_trials_z(:,tidx);


        % Trial-wise HGA difference
        D = x_t - B;


        D = D(isfinite(D));


        if numel(D) < 20
            continue;
        end


        % ================================================================
        % ONE-SIDED ONE-SAMPLE T-TEST
        %
        % H0: mean(D) <= 0
        % H1: mean(D) > 0
        % ================================================================

        [~,p_ttest(ii,kk)] = ...
            ttest( ...
                D, ...
                0, ...
                'Tail','right');

    end


    if mod(ii,20)==0 || ii==n_chan

        fprintf( ...
            'Finished %d / %d channels\n', ...
            ii,n_chan);

    end

end


%% ========================================================================
%% 8. GLOBAL BH-FDR ACROSS ELECTRODE x TIME
%% ========================================================================
%
% All valid electrode x time P values from this participant are corrected
% together.
%
%% ========================================================================

fprintf('\n============================================================\n');
fprintf('GLOBAL BH-FDR CORRECTION\n');
fprintf('============================================================\n');


[p_fdr,h_fdr] = ...
    fdr_bh_matrix( ...
        p_ttest, ...
        fdr_q);


fprintf('Total valid tests      = %d\n', ...
    sum(isfinite(p_ttest(:))));

fprintf('FDR-significant points = %d\n', ...
    sum(h_fdr(:)));


%% ========================================================================
%% 9. IDENTIFY >=100-ms CONTIGUOUS SIGNIFICANT RESPONSES
%% ========================================================================

responsive_mask = false(n_chan,1);


% First qualifying cluster
latency_ms = nan(n_chan,1);

first_offset_ms = nan(n_chan,1);

first_duration_ms = nan(n_chan,1);


% Number of >=100-ms significant clusters per electrode
n_qual_clusters = zeros(n_chan,1);


% Keep ONLY FDR-significant points belonging to >=100-ms clusters
qualifying_mask = false(n_chan,n_time);


% Save all qualifying cluster starts/ends
cluster_onsets_all = cell(n_chan,1);

cluster_offsets_all = cell(n_chan,1);


fprintf('\n============================================================\n');
fprintf('SEARCHING >= %d ms CONTIGUOUS RESPONSES\n',min_contig_ms);
fprintf('============================================================\n');


for ii = 1:n_chan

    h = logical(h_fdr(ii,:));


    [qual_mask,starts_ms,ends_ms] = ...
        find_qualifying_clusters( ...
            h, ...
            search_t, ...
            min_contig_len);


    qualifying_mask(ii,:) = qual_mask;


    cluster_onsets_all{ii} = starts_ms;

    cluster_offsets_all{ii} = ends_ms;


    n_qual_clusters(ii) = numel(starts_ms);


    if ~isempty(starts_ms)

        responsive_mask(ii) = true;


        % ---------------------------------------------------------------
        % First qualifying significant cluster defines latency
        % ---------------------------------------------------------------

        latency_ms(ii) = starts_ms(1);

        first_offset_ms(ii) = ends_ms(1);

        first_duration_ms(ii) = ...
            ends_ms(1) - ...
            starts_ms(1) + ...
            dt_ms;

    end

end


%% ========================================================================
%% 10. RESPONSIVE CHANNEL NUMBERS
%% ========================================================================

responsive_chans = ...
    chan_list(responsive_mask).';


% Column vector, convenient for later analysis
responsive_chans_col = ...
    responsive_chans(:);


% Row vector
responsive_chans_row = ...
    responsive_chans(:).';


% Responsive-channel indices within chan_list
responsive_idx = ...
    find(responsive_mask);


% Responsive latency
latency_ms_resp = ...
    latency_ms(responsive_mask);


first_offset_ms_resp = ...
    first_offset_ms(responsive_mask);


first_duration_ms_resp = ...
    first_duration_ms(responsive_mask);


n_qual_clusters_resp = ...
    n_qual_clusters(responsive_mask);


fprintf('\n============================================================\n');
fprintf('FINAL RESPONSIVE CONTACTS\n');
fprintf('============================================================\n');

fprintf( ...
    'Responsive contacts = %d / %d\n', ...
    numel(responsive_chans), ...
    n_chan);


disp('Responsive channel numbers:');

disp(responsive_chans_col);


%% ========================================================================
%% 11. CHANNEL LABELS
%% ========================================================================

responsive_labels = ...
    cell(numel(responsive_chans),1);


for jj = 1:numel(responsive_chans)

    nchan = responsive_chans(jj);

    try

        responsive_labels{jj} = ...
            ALLEEG(1).chanlocs(nchan).labels;

    catch

        responsive_labels{jj} = ...
            ['Chan_' num2str(nchan)];

    end

end


%% ========================================================================
%% 12. RESPONSIVE INFO MATRIX
%% ========================================================================
%
% Columns:
%
%   1 = Channel number
%   2 = First significant-cluster onset (ms)
%   3 = First significant-cluster offset (ms)
%   4 = First significant-cluster duration (ms)
%   5 = Number of qualifying >=100-ms clusters
%
%% ========================================================================

responsive_info = [ ...
    responsive_chans_col, ...
    latency_ms_resp, ...
    first_offset_ms_resp, ...
    first_duration_ms_resp, ...
    n_qual_clusters_resp];


%% ========================================================================
%% 13. RESPONSIVE SUMMARY TABLE
%% ========================================================================

responsive_table = table( ...
    responsive_chans_col, ...
    responsive_labels, ...
    latency_ms_resp, ...
    first_offset_ms_resp, ...
    first_duration_ms_resp, ...
    n_qual_clusters_resp, ...
    ...
    'VariableNames',{ ...
    'Channel', ...
    'Label', ...
    'FirstOnset_ms', ...
    'FirstOffset_ms', ...
    'FirstDuration_ms', ...
    'N_QualifyingClusters'});


disp(responsive_table);


%% ========================================================================
%% 14. SAVE RESPONSIVE RESULTS
%% ========================================================================

result_save_name = ...
    fullfile( ...
        saveDir, ...
        sprintf( ...
        '%s_final_responsive_q005_100ms.mat', ...
        subjName));


save( ...
    result_save_name, ...
    ...
    'responsive_chans', ...
    'responsive_chans_col', ...
    'responsive_chans_row', ...
    'responsive_info', ...
    'responsive_table', ...
    ...
    'responsive_mask', ...
    'responsive_idx', ...
    ...
    'latency_ms', ...
    'latency_ms_resp', ...
    'first_offset_ms', ...
    'first_offset_ms_resp', ...
    'first_duration_ms', ...
    'first_duration_ms_resp', ...
    ...
    'n_qual_clusters', ...
    'n_qual_clusters_resp', ...
    'cluster_onsets_all', ...
    'cluster_offsets_all', ...
    ...
    'p_ttest', ...
    'p_fdr', ...
    'h_fdr', ...
    'qualifying_mask', ...
    ...
    'baseline_win', ...
    'responsive_search_win', ...
    'fdr_q', ...
    'min_contig_ms', ...
    'min_contig_len', ...
    'dt_ms', ...
    ...
    'chan_list', ...
    'chans_to_remove', ...
    'n_trials_all');


fprintf('\nSaved MAT result to:\n%s\n', ...
    result_save_name);


% ------------------------------------------------------------------------
% Also save simple responsive channel vector separately
% ------------------------------------------------------------------------

save( ...
    fullfile( ...
        saveDir, ...
        sprintf('%s_responsive_contacts.mat',subjName)), ...
    'responsive_chans_col');


% ------------------------------------------------------------------------
% Save numeric matrix separately
% ------------------------------------------------------------------------

save( ...
    fullfile( ...
        saveDir, ...
        sprintf('%s_responsive_info.mat',subjName)), ...
    'responsive_info');


% ------------------------------------------------------------------------
% Save human-readable CSV
% ------------------------------------------------------------------------

writetable( ...
    responsive_table, ...
    fullfile( ...
        saveDir, ...
        sprintf('%s_responsive_contacts.csv',subjName)));


%% ========================================================================
%% 15. PLOT ALL FINAL RESPONSIVE CONTACTS
%% ========================================================================
%
% Plot:
%
%   Blue curve       = mean HGA across all trials
%   Blue shading     = SEM
%   Grey top line    = all FDR-significant points (q < 0.05)
%   Red top line     = significant segments lasting >=100 ms
%   Red dotted line  = onset of FIRST qualifying cluster = response latency
%
%% ========================================================================

fprintf('\n============================================================\n');
fprintf('PLOTTING FINAL RESPONSIVE CONTACTS\n');
fprintf('============================================================\n');


figHandles = gobjects(0);

pageIdx = 0;


for jj = 1:numel(responsive_chans)

    nchan = responsive_chans(jj);


    % Find position within chan_list
    ii = find(chan_list == nchan,1);


    % --------------------------------------------------------------------
    % Collect all trials
    % --------------------------------------------------------------------

    all_trials_z = ...
        collect_trials_zscore( ...
            ALLEEG, ...
            cond_list, ...
            nchan, ...
            t, ...
            base_indices);


    n_trials_total = size(all_trials_z,1);


    % --------------------------------------------------------------------
    % Mean and SEM
    % --------------------------------------------------------------------

    Ave = mean( ...
        all_trials_z, ...
        1, ...
        'omitnan');


    SE = std( ...
        all_trials_z, ...
        0, ...
        1, ...
        'omitnan') ./ ...
        sqrt(n_trials_total);


    % --------------------------------------------------------------------
    % Full-time significance masks
    % --------------------------------------------------------------------

    sig_full = false(size(t));

    sig_full(search_indices) = ...
        h_fdr(ii,:);


    qualifying_full = false(size(t));

    qualifying_full(search_indices) = ...
        qualifying_mask(ii,:);


    % First qualifying onset
    sig_onset_time = ...
        latency_ms(ii);


    % --------------------------------------------------------------------
    % Figure pagination: 30 contacts / page
    % --------------------------------------------------------------------

    plot_idx = ...
        mod(jj-1,n_plot_per_page) + 1;


    if plot_idx == 1

        pageIdx = pageIdx + 1;


        hFig = figure( ...
            'Position',[50 50 1800 1000], ...
            'Color','w');


        figHandles(pageIdx) = hFig;

    end


    subplot(6,5,plot_idx);


    %% ==================================================================
    % Mean ± SEM
    %% ==================================================================

    hold on;


    x_fill = [t fliplr(t)];

    y_fill = [ ...
        Ave-SE, ...
        fliplr(Ave+SE)];


    h_fill = fill( ...
        x_fill, ...
        y_fill, ...
        [0.75 0.85 1], ...
        'EdgeColor','none', ...
        'FaceAlpha',0.35);


    h_mean = plot( ...
        t, ...
        Ave, ...
        'b', ...
        'LineWidth',1.2);


    %% ==================================================================
    % Baseline zero line / stimulus onset
    %% ==================================================================

    plot( ...
        t, ...
        zeros(size(t)), ...
        'k--', ...
        'LineWidth',0.5);


    xline( ...
        0, ...
        'Color',[0.65 0.65 0.65], ...
        'LineWidth',1);


    %% ==================================================================
    % Statistical significance
    %% ==================================================================

    % ---------------------------------------------------------------
    % Grey:
    % every FDR-significant point
    % ---------------------------------------------------------------

    significance_y_level = ...
        plot_ylim(2) - 0.18;


    sig_y = ...
        ones(size(t)) .* significance_y_level;


    sig_y(~sig_full) = NaN;


    plot( ...
        t, ...
        sig_y, ...
        'Color',[0.65 0.65 0.65], ...
        'LineWidth',2);


    % ---------------------------------------------------------------
    % Red:
    % ONLY significant segments lasting >=100 ms
    % ---------------------------------------------------------------

    qualifying_y_level = ...
        plot_ylim(2) - 0.08;


    qual_y = ...
        ones(size(t)) .* qualifying_y_level;


    qual_y(~qualifying_full) = NaN;


    plot( ...
        t, ...
        qual_y, ...
        'r', ...
        'LineWidth',2.5);


    %% ==================================================================
    % First significant-cluster onset / latency
    %% ==================================================================

    if ~isnan(sig_onset_time)


        plot( ...
            [sig_onset_time sig_onset_time], ...
            [0 qualifying_y_level], ...
            'r:', ...
            'LineWidth',1);


        text( ...
            sig_onset_time - 8, ...
            qualifying_y_level - 0.15, ...
            sprintf('%.0f ms',sig_onset_time), ...
            'FontSize',8, ...
            'Color','r', ...
            'HorizontalAlignment','right', ...
            'VerticalAlignment','top');

    end


    %% ==================================================================
    % Channel label
    %% ==================================================================

    try

        chan_label = ...
            ALLEEG(1).chanlocs(nchan).labels;

    catch

        chan_label = ...
            ['Chan ' num2str(nchan)];

    end


    title( ...
        sprintf('%s (%d)',chan_label,nchan), ...
        'FontSize',11, ...
        'FontWeight','bold', ...
        'Interpreter','none');


    %% ==================================================================
    % Axes
    %% ==================================================================

    if plot_idx > 25

        xlabel( ...
            'Time (ms)', ...
            'FontSize',9);

    end


    if ismember(plot_idx,[1 6 11 16 21 26])

        ylabel( ...
            'HGA (z-score)', ...
            'FontSize',9);

    end


    xlim(plot_xlim);

    ylim(plot_ylim);


    set( ...
        gca, ...
        'TickDir','out', ...
        'FontSize',9);


    box off;


    fprintf( ...
        '[Plot] %d / %d | Chan %d | latency = %.1f ms\n', ...
        jj, ...
        numel(responsive_chans), ...
        nchan, ...
        sig_onset_time);

end


%% ========================================================================
%% 16. SAVE RESPONSE FIGURES
%% ========================================================================

for p = 1:numel(figHandles)

    if ~isgraphics(figHandles(p))
        continue;
    end


    pngName = ...
        sprintf( ...
            '%s%02d.png', ...
            imgPrefix, ...
            p);


    exportgraphics( ...
        figHandles(p), ...
        fullfile(saveDir,pngName), ...
        'Resolution',600);

end


fprintf('\nAll response figures saved to:\n%s\n', ...
    saveDir);

%% ========================================================================
%% 17. PLOT ALL NON-RESPONSIVE CONTACTS
%% ========================================================================
%
% Non-responsive definition:
%
% No >=100-ms contiguous FDR-significant positive HGA response
% within 0-800 ms.
%
% Plot:
%
%   Blue curve       = mean HGA across all trials
%   Blue shading     = SEM
%   Grey top line    = all FDR-significant points (q < 0.05)
%
% No red qualifying line and no latency are shown because these contacts
% do not contain any >=100-ms qualifying response.
%
%% ========================================================================

fprintf('\n============================================================\n');
fprintf('PLOTTING NON-RESPONSIVE CONTACTS\n');
fprintf('============================================================\n');


%% ------------------------------------------------------------------------
% Non-responsive channel numbers
%% ------------------------------------------------------------------------

nonresponsive_mask = ~responsive_mask;

nonresponsive_chans = ...
    chan_list(nonresponsive_mask).';

nonresponsive_chans_col = ...
    nonresponsive_chans(:);


fprintf( ...
    'Non-responsive contacts = %d / %d\n', ...
    numel(nonresponsive_chans), ...
    numel(chan_list));


disp('Non-responsive channel numbers:');

disp(nonresponsive_chans_col);


%% ------------------------------------------------------------------------
% Separate output folder
%% ------------------------------------------------------------------------

saveDir_nonresp = ...
    fullfile( ...
        saveDir, ...
        'nonresponsive_contacts');


if ~exist(saveDir_nonresp,'dir')
    mkdir(saveDir_nonresp);
end


imgPrefix_nonresp = ...
    [subjName '_52words_Nonresponsive_Page_'];


%% ------------------------------------------------------------------------
% Figure pagination
%% ------------------------------------------------------------------------

figHandles_nonresp = gobjects(0);

pageIdx_nonresp = 0;


for jj = 1:numel(nonresponsive_chans)

    nchan = nonresponsive_chans(jj);


    % --------------------------------------------------------------------
    % Find location of channel within chan_list
    % --------------------------------------------------------------------

    ii = find(chan_list == nchan,1);


    % --------------------------------------------------------------------
    % Collect all trials
    % --------------------------------------------------------------------

    all_trials_z = ...
        collect_trials_zscore( ...
            ALLEEG, ...
            cond_list, ...
            nchan, ...
            t, ...
            base_indices);


    n_trials_total = ...
        size(all_trials_z,1);


    % --------------------------------------------------------------------
    % Mean and SEM
    % --------------------------------------------------------------------

    Ave = mean( ...
        all_trials_z, ...
        1, ...
        'omitnan');


    SE = std( ...
        all_trials_z, ...
        0, ...
        1, ...
        'omitnan') ./ ...
        sqrt(n_trials_total);


    % --------------------------------------------------------------------
    % Full-time FDR significance mask
    %
    % NOTE:
    % non-responsive contacts may still contain short significant periods,
    % but none lasts >=100 ms.
    % --------------------------------------------------------------------

    sig_full = false(size(t));

    sig_full(search_indices) = ...
        h_fdr(ii,:);


    % --------------------------------------------------------------------
    % Pagination: 30 contacts per page
    % --------------------------------------------------------------------

    plot_idx = ...
        mod(jj-1,n_plot_per_page) + 1;


    if plot_idx == 1

        pageIdx_nonresp = ...
            pageIdx_nonresp + 1;


        hFig = figure( ...
            'Position',[50 50 1800 1000], ...
            'Color','w');


        figHandles_nonresp(pageIdx_nonresp) = ...
            hFig;

    end


    subplot(6,5,plot_idx);


    hold on;


    %% ==================================================================
    % Mean ± SEM
    %% ==================================================================

    x_fill = ...
        [t fliplr(t)];


    y_fill = ...
        [ ...
        Ave-SE, ...
        fliplr(Ave+SE) ...
        ];


    fill( ...
        x_fill, ...
        y_fill, ...
        [0.75 0.85 1], ...
        'EdgeColor','none', ...
        'FaceAlpha',0.35);


    plot( ...
        t, ...
        Ave, ...
        'b', ...
        'LineWidth',1.2);


    %% ==================================================================
    % Baseline and stimulus onset
    %% ==================================================================

    plot( ...
        t, ...
        zeros(size(t)), ...
        'k--', ...
        'LineWidth',0.5);


    xline( ...
        0, ...
        'Color',[0.65 0.65 0.65], ...
        'LineWidth',1);


    %% ==================================================================
    % FDR-significant points
    %
    % Grey line:
    % all q<0.05 points, even though they fail the >=100-ms criterion
    %% ==================================================================

    significance_y_level = ...
        plot_ylim(2) - 0.12;


    if any(sig_full)

        sig_y = ...
            ones(size(t)) .* ...
            significance_y_level;


        sig_y(~sig_full) = NaN;


        plot( ...
            t, ...
            sig_y, ...
            'Color',[0.60 0.60 0.60], ...
            'LineWidth',2);

    end


    %% ==================================================================
    % Channel label
    %% ==================================================================

    try

        chan_label = ...
            ALLEEG(1).chanlocs(nchan).labels;

    catch

        chan_label = ...
            ['Chan ' num2str(nchan)];

    end


    title( ...
        sprintf('%s (%d)',chan_label,nchan), ...
        'FontSize',11, ...
        'FontWeight','bold', ...
        'Interpreter','none');


    %% ==================================================================
    % Axes
    %% ==================================================================

    if plot_idx > 25

        xlabel( ...
            'Time (ms)', ...
            'FontSize',9);

    end


    if ismember(plot_idx,[1 6 11 16 21 26])

        ylabel( ...
            'HGA (z-score)', ...
            'FontSize',9);

    end


    xlim(plot_xlim);

    ylim(plot_ylim);


    set( ...
        gca, ...
        'TickDir','out', ...
        'FontSize',9);


    box off;


    fprintf( ...
        '[Non-responsive Plot] %d / %d | Chan %d\n', ...
        jj, ...
        numel(nonresponsive_chans), ...
        nchan);

end


%% ========================================================================
%% 18. SAVE NON-RESPONSIVE FIGURES
%% ========================================================================

for p = 1:numel(figHandles_nonresp)

    if ~isgraphics(figHandles_nonresp(p))
        continue;
    end


    pngName = ...
        sprintf( ...
            '%s%02d.png', ...
            imgPrefix_nonresp, ...
            p);


    exportgraphics( ...
        figHandles_nonresp(p), ...
        fullfile( ...
            saveDir_nonresp, ...
            pngName), ...
        'Resolution',600);

end


fprintf('\nAll non-responsive figures saved to:\n%s\n', ...
    saveDir_nonresp);


%% ========================================================================
%% 19. SAVE NON-RESPONSIVE CHANNEL NUMBERS
%% ========================================================================

save( ...
    fullfile( ...
        saveDir, ...
        sprintf( ...
            '%s_nonresponsive_contacts.mat', ...
            subjName)), ...
    'nonresponsive_chans_col');


%% ------------------------------------------------------------------------
% Labels
%% ------------------------------------------------------------------------

nonresponsive_labels = ...
    cell(numel(nonresponsive_chans),1);


for jj = 1:numel(nonresponsive_chans)

    nchan = ...
        nonresponsive_chans(jj);


    try

        nonresponsive_labels{jj} = ...
            ALLEEG(1).chanlocs(nchan).labels;

    catch

        nonresponsive_labels{jj} = ...
            ['Chan_' num2str(nchan)];

    end

end


%% ------------------------------------------------------------------------
% Table
%% ------------------------------------------------------------------------

nonresponsive_table = table( ...
    nonresponsive_chans_col, ...
    nonresponsive_labels, ...
    ...
    'VariableNames',{ ...
    'Channel', ...
    'Label'});


writetable( ...
    nonresponsive_table, ...
    fullfile( ...
        saveDir, ...
        sprintf( ...
            '%s_nonresponsive_contacts.csv', ...
            subjName)));


fprintf('\n============================================================\n');
fprintf('RESPONSIVE / NON-RESPONSIVE SUMMARY\n');
fprintf('============================================================\n');

fprintf( ...
    'Responsive contacts     = %d\n', ...
    numel(responsive_chans));

fprintf( ...
    'Non-responsive contacts = %d\n', ...
    numel(nonresponsive_chans));

fprintf( ...
    'Total analysed contacts = %d\n', ...
    numel(chan_list));

fprintf( ...
    'Check: %d + %d = %d\n', ...
    numel(responsive_chans), ...
    numel(nonresponsive_chans), ...
    numel(responsive_chans) + ...
    numel(nonresponsive_chans));

%% ========================================================================
%% FINAL SUMMARY
%% ========================================================================

fprintf('\n============================================================\n');
fprintf('FINAL SUMMARY\n');
fprintf('============================================================\n');

fprintf('Subject                  : %s\n',subjName);

fprintf('Analysed contacts        : %d\n', ...
    numel(chan_list));

fprintf('Responsive contacts      : %d\n', ...
    numel(responsive_chans));

fprintf('Baseline                 : %.0f to %.0f ms\n', ...
    baseline_win(1), ...
    baseline_win(2));

fprintf('Responsive search        : %.0f to %.0f ms\n', ...
    responsive_search_win(1), ...
    responsive_search_win(2));

fprintf('Test                     : one-sided one-sample t-test\n');

fprintf('BH-FDR                   : q < %.2f\n',fdr_q);

fprintf('Minimum contiguous       : >= %d ms\n', ...
    min_contig_ms);

fprintf('MAT result               : %s\n', ...
    result_save_name);


%% ========================================================================
%% LOCAL FUNCTION 1
%% Collect all trials and perform trial-wise baseline z-score
%% ========================================================================

function all_trials_z = ...
    collect_trials_zscore( ...
        ALLEEG, ...
        cond_list, ...
        nchan, ...
        t, ...
        base_indices)


    all_trials_z = [];


    for idx_condition = cond_list


        raw_data = double( ...
            squeeze( ...
                ALLEEG(idx_condition).data( ...
                    nchan,:,:)));


        % ----------------------------------------------------------------
        % Ensure [trial x time]
        % ----------------------------------------------------------------

        if isvector(raw_data)

            raw_data = raw_data(:)';

        end


        if size(raw_data,2) ~= numel(t)

            raw_data = raw_data';

        end


        if size(raw_data,2) ~= numel(t)

            error( ...
                'Channel %d condition %d: data dimensions do not match t.', ...
                nchan, ...
                idx_condition);

        end


        % ----------------------------------------------------------------
        % Trial-wise baseline normalization
        % ----------------------------------------------------------------

        baseline_data = ...
            raw_data(:,base_indices);


        mu = mean( ...
            baseline_data, ...
            2, ...
            'omitnan');


        sd = std( ...
            baseline_data, ...
            0, ...
            2, ...
            'omitnan');


        sd( ...
            ~isfinite(sd) | ...
            sd < eps) = eps;


        z_data = ...
            (raw_data-mu) ./ sd;


        % ----------------------------------------------------------------
        % Pool all words / conditions
        % ----------------------------------------------------------------

        all_trials_z = ...
            [all_trials_z;z_data];

    end

end


%% ========================================================================
%% LOCAL FUNCTION 2
%% Global Benjamini-Hochberg FDR
%% ========================================================================

function [p_adj_mat,h_mat] = ...
    fdr_bh_matrix( ...
        p_mat, ...
        q)


    original_size = size(p_mat);


    pvec = p_mat(:);


    valid = isfinite(pvec);


    pv = pvec(valid);


    p_adj_vec = nan(size(pvec));

    h_vec = false(size(pvec));


    if isempty(pv)

        p_adj_mat = ...
            reshape( ...
                p_adj_vec, ...
                original_size);


        h_mat = ...
            reshape( ...
                h_vec, ...
                original_size);

        return;

    end


    m = numel(pv);


    % --------------------------------------------------------------------
    % Sort P values
    % --------------------------------------------------------------------

    [ps,order] = sort(pv);


    rank = (1:m)';


    % --------------------------------------------------------------------
    % BH critical line
    % --------------------------------------------------------------------

    critical = ...
        (rank./m).*q;


    below = ...
        ps <= critical;


    h_sorted = false(m,1);


    if any(below)

        kmax = ...
            find( ...
                below, ...
                1, ...
                'last');


        h_sorted(1:kmax) = true;

    end


    % --------------------------------------------------------------------
    % BH-adjusted P values
    % --------------------------------------------------------------------

    p_adj_sorted = ...
        ps .* m ./ rank;


    p_adj_sorted = ...
        flipud( ...
            cummin( ...
                flipud(p_adj_sorted)));


    p_adj_sorted = ...
        min(p_adj_sorted,1);


    % --------------------------------------------------------------------
    % Return to original order
    % --------------------------------------------------------------------

    p_adj_valid = nan(m,1);

    h_valid = false(m,1);


    p_adj_valid(order) = ...
        p_adj_sorted;


    h_valid(order) = ...
        h_sorted;


    p_adj_vec(valid) = ...
        p_adj_valid;


    h_vec(valid) = ...
        h_valid;


    p_adj_mat = ...
        reshape( ...
            p_adj_vec, ...
            original_size);


    h_mat = ...
        reshape( ...
            h_vec, ...
            original_size);

end


%% ========================================================================
%% LOCAL FUNCTION 3
%% Find ALL significant clusters lasting >= min_len
%% ========================================================================

function [qualifying_mask,starts_ms,ends_ms] = ...
    find_qualifying_clusters( ...
        h_time, ...
        t_win, ...
        min_len)


    h_time = logical(h_time);


    qualifying_mask = false(size(h_time));


    starts_ms = [];

    ends_ms = [];


    if ~any(h_time)
        return;
    end


    % --------------------------------------------------------------------
    % Find contiguous runs of TRUE values
    % --------------------------------------------------------------------

    padded = ...
        [false h_time false];


    transitions = ...
        diff(padded);


    starts = ...
        find(transitions == 1);


    ends = ...
        find(transitions == -1) - 1;


    run_lengths = ...
        ends-starts+1;


    % --------------------------------------------------------------------
    % Keep only runs >= min_len
    % --------------------------------------------------------------------

    good_runs = ...
        find(run_lengths >= min_len);


    if isempty(good_runs)
        return;
    end


    for kk = 1:numel(good_runs)

        r = good_runs(kk);


        s = starts(r);

        e = ends(r);


        qualifying_mask(s:e) = true;


        starts_ms(end+1,1) = ...
            t_win(s);


        ends_ms(end+1,1) = ...
            t_win(e);

    end

end
