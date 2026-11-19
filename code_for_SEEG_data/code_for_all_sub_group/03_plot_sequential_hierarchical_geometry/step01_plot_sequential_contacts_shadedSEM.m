%% 01_plot_sequential_units.m
% =========================================================================
% Sequential representation analysis for the MAIN factor-controlled unit set
%
% INPUT
% -----
% <repository>/code_for_SEEG_data/
%   code_for_all_sub_group\02_classify_units\
%   main_factorControlled_2consecutive_with_downstream\
%   downstream_analysis_inputs_200ms_50ms\
%       01_sequential_strength\
%           allResponsive_P_strength_16windows.mat
%           allResponsive_S_strength_16windows.mat
%
% Expected columns in both variables:
%   1  Subject
%   2  Contact
%   3  P significant (0/1)
%   4  S significant (0/1)
%   5  Unit class: 0=None, 1=P-only, 2=S-only, 3=Shared
%   6:21 representation strengths for 16 windows (50:50:800 ms)
%
% OUTPUT
% ------
% A single clean subfolder under the CURRENT plotting folder:
%
%   03_plot_sequential_hierarchical_geometry\
%       results_01_sequential\
%           Sequential_main_combinedHeatmap_lines.pdf
%           Sequential_main_combinedHeatmap_lines.png
%           Sequential_main_combinedHeatmap_lines_0to600ms.pdf
%           Sequential_main_combinedHeatmap_lines_0to600ms.png
%           Sequential_supp_separate_P_S_heatmaps.pdf
%           Sequential_supp_separate_P_S_heatmaps.png
%           Sequential_supp_separate_P_S_heatmaps_0to600ms.pdf
%           Sequential_supp_separate_P_S_heatmaps_0to600ms.png
%           Sequential_participant_level_statistics.csv
%
% MAIN FIGURE
% -----------
% Three rows: P-only / S-only / Shared
%
% Left:
%   combined P-vs-S heatmap. At each contact x time cell, only the stronger
%   POSITIVE representation is shown:
%
%       P stronger -> +P_z  (red)
%       S stronger -> -S_z  (blue)
%       neither positive -> 0 (white)
%
% Right:
%   participant-weighted P and S trajectories (mean +/- SEM).
%   Contacts are first averaged WITHIN participant, then participants are
%   averaged at group level.
%   The current display uses row-specific y-axis ranges:
%       P-only  = automatic
%       S-only  = automatic
%       Shared  = automatic
%
% Significance markers (optional; participant-first):
%   Contacts are averaged within participant before inference.
%   Colored stars: one-sided Wilcoxon signed-rank for P>0 / S>0.
%   Black stars: paired two-sided Wilcoxon signed-rank for P versus S.
%
%   Correction is selectable. The current default is a participant-level
%   sign-flip cluster permutation across the 16-window trajectory:
%     - P/S >0: maximum cluster mass is taken jointly across P and S.
%     - P vs S: two-sided cluster permutation on the paired P-S trajectory.
%   Holm, Bonferroni, FDR, and uncorrected alternatives remain available.
%
%   The 0-600 ms display reuses the full 16-window corrected results.
%   Because unit classes are defined from the same dataset, these annotations
%   are descriptive/conditional rather than independent confirmation.
%
% SUPPLEMENTARY FIGURE
% --------------------
% P and S heatmaps are plotted separately, using the SAME contact order as
% the combined main heatmap.
%
% By default only positive z values are shown in these heatmaps because the
% unit phenotype is based on positive representational evidence.
%
% =========================================================================

clearvars;
close all;
clc;

%% ============================ USER SETTINGS =============================

% ------------------------------- Paths -----------------------------------
script_dir = fileparts(mfilename('fullpath'));
input_root = fullfile(fileparts(script_dir), '02_classify_units', ...
    'main_factorControlled_2consecutive_with_downstream', ...
    'downstream_analysis_inputs_200ms_50ms');

plot_root = script_dir;

save_dir = fullfile(plot_root,'results_01_sequential_shadedSEM');

if ~isfolder(save_dir)
    mkdir(save_dir);
end

% ---------------------------- Input files --------------------------------
P_file = fullfile( ...
    input_root, ...
    '01_sequential_strength', ...
    'allResponsive_P_strength_16windows.mat');

S_file = fullfile( ...
    input_root, ...
    '01_sequential_strength', ...
    'allResponsive_S_strength_16windows.mat');

assert(isfile(P_file), ...
    'Cannot find P-strength file:\n%s',P_file);

assert(isfile(S_file), ...
    'Cannot find S-strength file:\n%s',S_file);

% ---------------------------- Time settings ------------------------------
time_points = 50:50:800;       % 16 window centers
nwin = numel(time_points);

% Additional display-only version ending at 600 ms.
% The original 0-800 ms figures are retained.
plot_end_600 = 600;
time_mask_600 = time_points <= plot_end_600;
time_points_600 = time_points(time_mask_600);

% ---------------------------- Class settings -----------------------------
% Unit coding from the downstream export:
%   0=None, 1=P-only, 2=S-only, 3=Shared
plot_class_codes = [1 2 3];

class_titles = { ...
    'Phonological', ...
    'Semantic', ...
    'Shared'};

% ------------------------------ Colors ----------------------------------
P_color = [186 48 49]/256;       % red
S_color = [35 100 156]/256;      % blue

% Combined heatmap:
% negative = Semantic/blue, zero=white, positive=Phonological/red
combined_cmap = make_diverging_colormap( ...
    S_color,[1 1 1],P_color,256);

% Separate heatmaps:
% positive P = red, positive S = blue
P_cmap = make_white_to_color_colormap(P_color,256);
S_cmap = make_white_to_color_colormap(S_color,256);

% Fixed heatmap range for z values.
% Increase if many values saturate.
heatmap_zmax = 3;

% Only show positive P/S strength in separate heatmaps.
separate_heatmap_positive_only = true;

% ------------------------------- Sorting --------------------------------
% Do NOT latency-sort contacts because that can visually create diagonal
% temporal structure.
%
% P-only   : descending total positive P strength
% S-only   : descending total positive S strength
% Shared   : descending total positive joint P+S strength
sorting_mode = 'strength';

% ------------------------------ Line plot --------------------------------
% -------------------------------------------------------------------------
% Line-plot y-axis settings
% -------------------------------------------------------------------------
% true  = each unit class can use its own y-axis range
% false = all three rows share one common y-axis range
use_class_specific_line_ylim = true;

% Used only when use_class_specific_line_ylim = false.
% [] = choose one common y-axis automatically from group mean +/- SEM.
% Or manually set, e.g. [-0.5 2.5].
fixed_line_ylim = [];

% Class-specific settings:
%   P-only : [] -> automatic from P-only mean +/- SEM
%   S-only : fixed maximum = 1, as requested
%   Shared : [] -> automatic from Shared mean +/- SEM
%
% You can manually replace [] with [ymin ymax] if needed.
Ponly_line_ylim  = [];
Sonly_line_ylim  = [];
Shared_line_ylim = [];

% For automatic class-specific limits, the upper limit is rounded upward
% to this step so the axis remains visually clean.
auto_ylim_step = 0.5;

% Extra headroom for significance bars.
% These are used only when y-limits are chosen automatically.
line_ylim_lower_pad_frac = 0.10;
line_ylim_upper_pad_frac = 0.22;
line_ylim_min_lower = -0.20;

show_errorbars = true;

% -------------------------- Significance marks ---------------------------
% Participant-first inference is retained throughout:
%   contacts -> average within participant -> group-level signed-rank test.
%
% Two kinds of optional significance annotations are shown:
%
% (1) Colored stars: whether each representation is significantly > 0
%       red  = phonological > 0
%       blue = semantic > 0
%
% (2) Black stars: whether phonological and semantic strengths differ
%       paired two-sided Wilcoxon signed-rank test on P - S
%
% Display switches
show_zero_significance = true;
show_PvsS_significance = false;

% For ordinary point-wise corrections (Holm/Bonferroni/FDR):
%   - P/S >0: within each contact class, 16 P + 16 S tests are treated as
%     ONE family of 32 tests.
%   - P-vs-S: the 16 paired tests form one family.
%
% For cluster permutation:
%   - P/S >0: P and S are tested JOINTLY within each contact class by taking
%     the maximum positive cluster mass across BOTH P and S on every
%     permutation. This controls FWER across representation dimension + time.
%   - P-vs-S: a two-sided sign-flip cluster test is applied to P-S across
%     the full 16-window trajectory.
%
% Significance/correction method:
%
%   'cluster'     -> participant-level sign-flip cluster permutation
%                    across the full 16-window trajectory (recommended for
%                    these strongly overlapping 200-ms / 50-ms-step windows)
%   'holm'        -> Holm-Bonferroni FWER control
%   'bonferroni'  -> Bonferroni FWER control
%   'fdr'         -> Benjamini-Hochberg FDR
%   'none'        -> uncorrected nominal p-values
%
% The two types of annotation can use different methods.
sig_correction_zero = 'cluster';
sig_correction_PvsS = 'cluster';

% Final corrected significance threshold.
sig_alpha = 0.01;

% ---------------- Cluster-permutation settings ---------------------------
% Cluster-forming threshold is UNCORRECTED and is used only to define
% candidate temporal clusters. Cluster-level significance is then tested
% against the maximum-cluster permutation distribution.
cluster_forming_alpha = 0.01;

% Number of participant-level sign-flip permutations.
cluster_nperm = 5000;

% Fixed seed for exact reproducibility.
cluster_seed = 20260920;

% Plot significant cluster results as continuous horizontal bars rather
% than one star at every significant window.
%   'bar'   -> continuous bars spanning significant clusters
%   'stars' -> stars at each significant window
cluster_plot_style = 'bar';

% Marker styles used for non-cluster methods, or if cluster_plot_style='stars'
zero_sig_marker = '*';
PvsS_sig_marker = '*';

% IMPORTANT:
% Unit classes were phenotyped using the same full dataset, so the colored
% >0 annotations remain descriptive/conditional rather than an independent
% confirmation of the class-defining effect.
%
% The black P-vs-S test is more directly related to relative P/S dominance,
% but it is still based on the same full-data trajectories.
%
% The 0-600 ms figure uses the SAME corrected results from all 16 pre-defined
% windows. Cropping the display therefore does not change the correction.

% ------------------------------- Saving ---------------------------------
save_png = true;
save_pdf = true;

png_resolution = 600;

%% ============================== LOAD DATA ================================

P_raw = load_strength_dataset( ...
    P_file, ...
    {'allResponsive_P_strength', ...
     'allResponsive_P_strength_16windows'});

S_raw = load_strength_dataset( ...
    S_file, ...
    {'allResponsive_S_strength', ...
     'allResponsive_S_strength_16windows'});

P = standardize_strength_dataset(P_raw,nwin,'P');
S = standardize_strength_dataset(S_raw,nwin,'S');

%% ========================== ALIGN P AND S ROWS ===========================

% Build unique Subject + Contact keys.
P_key = strcat(P.Subject,"__",string(P.Contact));
S_key = strcat(S.Subject,"__",string(S.Contact));

assert(numel(unique(P_key))==height(P), ...
    'P dataset contains duplicate Subject+Contact rows.');

assert(numel(unique(S_key))==height(S), ...
    'S dataset contains duplicate Subject+Contact rows.');

[tf,locS] = ismember(P_key,S_key);

assert(all(tf), ...
    'Some P rows do not have a matching Subject+Contact in S.');

S = S(locS,:);

assert(all(P.Subject==S.Subject), ...
    'P/S subject ordering mismatch after alignment.');

assert(all(P.Contact==S.Contact), ...
    'P/S contact ordering mismatch after alignment.');

% Metadata should be identical.
assert(all(P.Psig==S.Psig), ...
    'P/S files disagree in P significance flag.');

assert(all(P.Ssig==S.Ssig), ...
    'P/S files disagree in S significance flag.');

assert(all(P.UnitClass==S.UnitClass), ...
    'P/S files disagree in unit class.');

% Extract numeric strengths.
Pz = P{:,compose("T%03d",time_points)};
Sz = S{:,compose("T%03d",time_points)};

assert(isequal(size(Pz),size(Sz)), ...
    'P/S strength matrices differ in size.');

assert(size(Pz,2)==nwin, ...
    'Expected %d windows, found %d.',nwin,size(Pz,2));

%% ========================== BASIC DIAGNOSTICS ============================

fprintf('\n============================================================\n');
fprintf('SEQUENTIAL INPUT SUMMARY\n');
fprintf('============================================================\n');

fprintf('Responsive contacts: %d\n',height(P));

for iclass = 1:numel(plot_class_codes)

    cls = plot_class_codes(iclass);

    use = P.UnitClass==cls;

    fprintf('%-14s: %d contacts | %d subjects\n', ...
        class_titles{iclass}, ...
        nnz(use), ...
        numel(unique(P.Subject(use))));
end

%% ======================== PREPARE CLASS DATA =============================

% Reproducible randomization for cluster-permutation inference.
if strcmpi(sig_correction_zero,'cluster') || ...
        strcmpi(sig_correction_PvsS,'cluster')
    rng(cluster_seed,'twister');
end


ClassData = struct();

all_line_values = [];

stats_table = table();

for iclass = 1:numel(plot_class_codes)

    cls = plot_class_codes(iclass);

    use = P.UnitClass==cls;

    idx_global = find(use);

    Pclass = Pz(use,:);
    Sclass = Sz(use,:);

    SubClass = P.Subject(use);
    ContactClass = P.Contact(use);

    % ---------------------------------------------------------------------
    % Contact sorting.
    % ---------------------------------------------------------------------
    switch lower(sorting_mode)

        case 'strength'

            if cls==1

                % P-only:
                % stronger overall P representation at the TOP
                sort_score = sum(max(Pclass,0),2);
                [~,ord] = sort(sort_score,'descend');

            elseif cls==2
        
                % S-only:
                % stronger overall S representation at the BOTTOM
                sort_score = sum(max(Sclass,0),2);
                [~,ord] = sort(sort_score,'ascend');

            elseif cls==3

                % Shared:
                % overall P-dominant contacts at the TOP,
                % overall S-dominant contacts at the BOTTOM.
                % This uses time-integrated P/S balance rather than transition latency.
                sort_score = ...
                    sum(max(Pclass,0),2) - ...
                    sum(max(Sclass,0),2);

                [~,ord] = sort(sort_score,'descend');

            else

                error('Unexpected class code.');

            end
        
        case 'subject_contact'

            % Neutral order: subject then contact.
            tmp = table(SubClass,ContactClass, ...
                'VariableNames',{'Subject','Contact'});

            [~,ord] = sortrows(tmp,{'Subject','Contact'});

        otherwise

            error('Unknown sorting_mode: %s',sorting_mode);
    end

    P_sorted = Pclass(ord,:);
    S_sorted = Sclass(ord,:);

    subject_sorted = SubClass(ord);
    contact_sorted = ContactClass(ord);
    global_idx_sorted = idx_global(ord);

    % ---------------------------------------------------------------------
    % Combined heatmap:
    % positive P -> positive/red
    % positive S -> negative/blue
    % ---------------------------------------------------------------------

    Ppos = max(P_sorted,0);
    Spos = max(S_sorted,0);

    Hcombined = zeros(size(Ppos));

    useP = Ppos>=Spos & Ppos>0;
    useS = Spos>Ppos & Spos>0;

    Hcombined(useP) = Ppos(useP);
    Hcombined(useS) = -Spos(useS);

    % Separate heatmaps.
    if separate_heatmap_positive_only
        Hpho = Ppos;
        Hsem = Spos;
    else
        Hpho = P_sorted;
        Hsem = S_sorted;
    end

    % ---------------------------------------------------------------------
    % Participant-weighted trajectories.
    % Average contacts within participant first.
    % ---------------------------------------------------------------------

    subjects_here = unique(SubClass,'stable');

    nsub = numel(subjects_here);

    Psub = nan(nsub,nwin);
    Ssub = nan(nsub,nwin);

    Ncontacts_per_subject = zeros(nsub,1);

    for isub = 1:nsub

        this_subject = subjects_here(isub);

        use_sub = SubClass==this_subject;

        Ncontacts_per_subject(isub) = nnz(use_sub);

        Psub(isub,:) = mean(Pclass(use_sub,:),1,'omitnan');
        Ssub(isub,:) = mean(Sclass(use_sub,:),1,'omitnan');
    end

    Pmean = mean(Psub,1,'omitnan');
    Smean = mean(Ssub,1,'omitnan');

    Psem = std(Psub,0,1,'omitnan') ./ sqrt(sum(isfinite(Psub),1));
    Ssem = std(Ssub,0,1,'omitnan') ./ sqrt(sum(isfinite(Ssub),1));

    all_line_values = [ ...
        all_line_values, ...
        Pmean-Psem,Pmean+Psem, ...
        Smean-Ssem,Smean+Ssem]; %#ok<AGROW>

    % ---------------------------------------------------------------------
    % PARTICIPANT-FIRST SIGNIFICANCE TESTS
    % ---------------------------------------------------------------------
    %
    % A. P > 0 and S > 0:
    %    one-sided Wilcoxon signed-rank across participant-level means.
    %
    % B. P versus S:
    %    paired two-sided Wilcoxon signed-rank across participants,
    %    applied to Psub(:,w) - Ssub(:,w).
    %
    % Multiple comparisons:
    %    - P/S >0: 32 tests (16 P + 16 S) corrected together within class.
    %    - P-vs-S: 16 tests corrected together within class.

    P_p = nan(1,nwin);
    S_p = nan(1,nwin);
    PvsS_p = nan(1,nwin);
    PminusS_mean = nan(1,nwin);

    for w = 1:nwin

        P_p(w) = one_sided_signrank_greater(Psub(:,w));
        S_p(w) = one_sided_signrank_greater(Ssub(:,w));

        [PvsS_p(w),PminusS_mean(w)] = ...
            paired_signrank_two_sided(Psub(:,w),Ssub(:,w));
    end

    % ---------------------------------------------------------------------
    % Corrected inference for P>0 and S>0.
    % ---------------------------------------------------------------------
    if strcmpi(sig_correction_zero,'cluster')

        [P_p_adj,S_p_adj,P_sig_plot,S_sig_plot] = ...
            cluster_perm_joint_positive( ...
            Psub,Ssub, ...
            cluster_nperm, ...
            cluster_forming_alpha, ...
            sig_alpha);

    else

        % Point-wise correction:
        % P and S are combined into one 32-test family within class.
        zero_p_family = [P_p S_p];
        zero_p_adj = adjust_pvalues( ...
            zero_p_family, ...
            sig_correction_zero);

        P_p_adj = zero_p_adj(1:nwin);
        S_p_adj = zero_p_adj(nwin+1:end);

        P_sig_plot = P_p_adj < sig_alpha;
        S_sig_plot = S_p_adj < sig_alpha;
    end

    % ---------------------------------------------------------------------
    % Corrected inference for paired P versus S.
    % ---------------------------------------------------------------------
    if strcmpi(sig_correction_PvsS,'cluster')

        [PvsS_p_adj,PvsS_sig_plot] = ...
            cluster_perm_paired_twosided( ...
            Psub,Ssub, ...
            cluster_nperm, ...
            cluster_forming_alpha, ...
            sig_alpha);

    else

        PvsS_p_adj = adjust_pvalues( ...
            PvsS_p, ...
            sig_correction_PvsS);

        PvsS_sig_plot = PvsS_p_adj < sig_alpha;
    end

    % ---------------------------------------------------------------------
    % Save class data.
    % ---------------------------------------------------------------------

    ClassData(iclass).class_code = cls;
    ClassData(iclass).title = class_titles{iclass};

    ClassData(iclass).P_sorted = P_sorted;
    ClassData(iclass).S_sorted = S_sorted;

    ClassData(iclass).Hcombined = Hcombined;
    ClassData(iclass).Hpho = Hpho;
    ClassData(iclass).Hsem = Hsem;

    ClassData(iclass).subject_sorted = subject_sorted;
    ClassData(iclass).contact_sorted = contact_sorted;
    ClassData(iclass).global_idx_sorted = global_idx_sorted;

    ClassData(iclass).subjects = subjects_here;

    ClassData(iclass).Psub = Psub;
    ClassData(iclass).Ssub = Ssub;

    ClassData(iclass).Pmean = Pmean;
    ClassData(iclass).Smean = Smean;

    ClassData(iclass).Psem = Psem;
    ClassData(iclass).Ssem = Ssem;

    ClassData(iclass).P_p = P_p;
    ClassData(iclass).S_p = S_p;

    ClassData(iclass).P_p_adj = P_p_adj;
    ClassData(iclass).S_p_adj = S_p_adj;

    ClassData(iclass).PvsS_p = PvsS_p;
    ClassData(iclass).PvsS_p_adj = PvsS_p_adj;
    ClassData(iclass).PminusS_mean = PminusS_mean;

    ClassData(iclass).P_sig_plot = P_sig_plot;
    ClassData(iclass).S_sig_plot = S_sig_plot;
    ClassData(iclass).PvsS_sig_plot = PvsS_sig_plot;

    ClassData(iclass).n_contacts = nnz(use);
    ClassData(iclass).n_subjects = nsub;

    % ---------------------------------------------------------------------
    % Statistics CSV rows.
    % ---------------------------------------------------------------------

    for w = 1:nwin

        row = table( ...
            string(class_titles{iclass}), ...
            cls, ...
            time_points(w), ...
            nnz(isfinite(Psub(:,w))), ...
            Pmean(w), ...
            Psem(w), ...
            P_p(w), ...
            P_p_adj(w), ...
            P_sig_plot(w), ...
            Smean(w), ...
            Ssem(w), ...
            S_p(w), ...
            S_p_adj(w), ...
            S_sig_plot(w), ...
            PminusS_mean(w), ...
            PvsS_p(w), ...
            PvsS_p_adj(w), ...
            PvsS_sig_plot(w), ...
            string(sig_correction_zero), ...
            string(sig_correction_PvsS), ...
            'VariableNames',{ ...
            'ContactClass', ...
            'ContactClassCode', ...
            'WindowCenter_ms', ...
            'N_participants', ...
            'P_mean_z', ...
            'P_SEM_z', ...
            'P_oneSided_p', ...
            'P_adjusted_p', ...
            'P_significantForPlot', ...
            'S_mean_z', ...
            'S_SEM_z', ...
            'S_oneSided_p', ...
            'S_adjusted_p', ...
            'S_significantForPlot', ...
            'PminusS_mean_z', ...
            'PvsS_twoSidedPaired_p', ...
            'PvsS_adjusted_p', ...
            'PvsS_significantForPlot', ...
            'ZeroTestCorrection', ...
            'PvsSCorrection'});

        stats_table = [stats_table;row]; %#ok<AGROW>
    end
end

%% ============================ LINE Y LIMITS ================================
%
% By default this version uses class-specific y-axis ranges:
%   P-only : automatic
%   S-only : automatic
%   Shared : automatic
%
% Automatic limits are derived from participant-weighted mean +/- SEM
% using the FULL 0-800 ms series, so the original and 0-600 ms figures
% use identical y limits within each row.

class_line_ylim = nan(3,2);

if use_class_specific_line_ylim

    manual_class_ylim = { ...
        Ponly_line_ylim, ...
        Sonly_line_ylim, ...
        Shared_line_ylim};

    for iclass = 1:3

        if ~isempty(manual_class_ylim{iclass})

            class_line_ylim(iclass,:) = manual_class_ylim{iclass};

        else

            D = ClassData(iclass);

            vals = [ ...
                D.Pmean-D.Psem, ...
                D.Pmean+D.Psem, ...
                D.Smean-D.Ssem, ...
                D.Smean+D.Ssem];

            vals = vals(isfinite(vals));

            if isempty(vals)

                class_line_ylim(iclass,:) = [-0.15 1];

            else

                ymin = min(vals);
                ymax = max(vals);

                yrange = ymax-ymin;

                if yrange<=eps
                    yrange = max(0.5,abs(ymax));
                end

                % Slightly more generous lower/upper padding so the
                % significance bars have extra space above the curves.
                lower_lim = min(line_ylim_min_lower, ...
                    ymin-line_ylim_lower_pad_frac*yrange);

                % Upper padding, then round upward to a clean increment.
                upper_raw = ymax + line_ylim_upper_pad_frac*yrange;
                upper_lim = ceil(upper_raw/auto_ylim_step)*auto_ylim_step;

                if upper_lim<=lower_lim
                    upper_lim = lower_lim + auto_ylim_step;
                end

                class_line_ylim(iclass,:) = [lower_lim upper_lim];
            end
        end
    end

    % Safety warning if the fixed S-only upper bound clips mean +/- SEM.
    Dsem = ClassData(2);
    sem_max = max([ ...
        Dsem.Pmean + Dsem.Psem, ...
        Dsem.Smean + Dsem.Ssem], ...
        [],'all','omitnan');

    if sem_max > class_line_ylim(2,2)
        warning([ ...
            'S-only fixed y-axis upper limit %.2f is below the observed ' ...
            'mean+SEM maximum %.3f. Increase Sonly_line_ylim if you do ' ...
            'not want clipping.'], ...
            class_line_ylim(2,2),sem_max);
    end

else

    % Original behavior: one common y-axis for all three rows.
    if isempty(fixed_line_ylim)

        vals = all_line_values(isfinite(all_line_values));

        if isempty(vals)

            line_ylim = [-1 1];

        else

            ymin = min(vals);
            ymax = max(vals);

            yrange = ymax-ymin;

            if yrange<=eps
                yrange = 1;
            end

            line_ylim = [ ...
                min(-0.15,ymin-0.08*yrange), ...
                ymax+0.22*yrange];
        end

    else

        line_ylim = fixed_line_ylim;
    end

    class_line_ylim = repmat(line_ylim,3,1);
end

fprintf('\n============================================================\n');
fprintf('LINE-PLOT Y LIMITS\n');
fprintf('============================================================\n');
for iclass = 1:3
    fprintf('%-14s: [%.3f, %.3f]\n', ...
        class_titles{iclass}, ...
        class_line_ylim(iclass,1), ...
        class_line_ylim(iclass,2));
end

%% ============================= MAIN FIGURE ===============================
%
% 3 rows:
%   left  = combined dominant-representation heatmap
%   right = participant-weighted trajectories with SEM and significance

h1 = figure( ...
    'Color','w', ...
    'Position',[100 100 315 260], ...
    'Renderer','painters');

tl = tiledlayout( ...
    3,5, ...
    'TileSpacing','compact', ...
    'Padding','compact');

for iclass = 1:3

    D = ClassData(iclass);

    % --------------------- Combined heatmap -------------------------------
    tile_heat = (iclass-1)*5+1;

    axH = nexttile(tile_heat,[1 2]);

    imagesc( ...
        axH, ...
        time_points, ...
        1:D.n_contacts, ...
        D.Hcombined);

    colormap(axH,combined_cmap);

    caxis(axH,[-heatmap_zmax heatmap_zmax]);

    set(axH, ...
        'FontSize',6, ...
        'FontName','Arial', ...
        'TickDir','out', ...
        'TickLength',[0 0], ...
        'LineWidth',0.5, ...
        'Box','on', ...
        'XColor','k', ...
        'YColor','k');

    ylabel(axH,'Contact number','FontSize',6);

    title(axH, ...
        sprintf('%s sEEG contacts (n = %d)', ...
        D.title,D.n_contacts), ...
        'FontSize',6, ...
        'FontWeight','normal');

    if iclass==3

        xlabel(axH,'Time (ms)','FontSize',6);

        xticks(axH,[0 200 400 600 800]);

    else

        xticks(axH,[0 200 400 600 800]);
        xticklabels(axH,[]);
    end

    % --------------------- Line plot -------------------------------------
    tile_line = (iclass-1)*5+3;

    axL = nexttile(tile_line,[1 3]);

    hold(axL,'on');

    if show_errorbars

        fill( ...
            axL, ...
            [time_points fliplr(time_points)], ...
            [D.Pmean+D.Psem fliplr(D.Pmean-D.Psem)], ...
            P_color, ...
            'FaceAlpha',0.18, ...
            'EdgeColor','none', ...
            'HandleVisibility','off');

        fill( ...
            axL, ...
            [time_points fliplr(time_points)], ...
            [D.Smean+D.Ssem fliplr(D.Smean-D.Ssem)], ...
            S_color, ...
            'FaceAlpha',0.18, ...
            'EdgeColor','none', ...
            'HandleVisibility','off');

        eP = plot( ...
            axL,time_points,D.Pmean,'-o', ...
            'Color',P_color, ...
            'MarkerEdgeColor',P_color, ...
            'MarkerFaceColor',P_color, ...
            'LineWidth',1.0, ...
            'MarkerSize',2.0, ...
            'DisplayName','Phonological');

        eS = plot( ...
            axL,time_points,D.Smean,'-o', ...
            'Color',S_color, ...
            'MarkerEdgeColor',S_color, ...
            'MarkerFaceColor',S_color, ...
            'LineWidth',1.0, ...
            'MarkerSize',2.0, ...
            'DisplayName','Semantic');

    else

        eP = plot( ...
            axL,time_points,D.Pmean,'-o', ...
            'Color',P_color, ...
            'MarkerEdgeColor',P_color, ...
            'MarkerFaceColor',P_color, ...
            'LineWidth',1.0, ...
            'MarkerSize',2.0, ...
            'DisplayName','Phonological');

        eS = plot( ...
            axL,time_points,D.Smean,'-o', ...
            'Color',S_color, ...
            'MarkerEdgeColor',S_color, ...
            'MarkerFaceColor',S_color, ...
            'LineWidth',1.0, ...
            'MarkerSize',2.0, ...
            'DisplayName','Semantic');
    end

    % yline( ...
    %     axL,0,'--', ...
    %     'Color',[0.65 0.65 0.65], ...
    %     'LineWidth',0.5, ...
    %     'HandleVisibility','off');

    xlim(axL,[0 800]);
    ylim(axL,class_line_ylim(iclass,:));

    xticks(axL,[0 200 400 600 800]);

    if iclass==3
        xlabel(axL,'Time (ms)','FontSize',6);
    else
        xticklabels(axL,[]);
    end

    ylabel(axL, ...
        {'Representation','index'}, ...
        'FontSize',6);

    title(axL, ...
        sprintf('%s | %d participants', ...
        D.title,D.n_subjects), ...
        'FontSize',6, ...
        'FontWeight','normal');

    set(axL, ...
        'FontSize',6, ...
        'FontName','Arial', ...
        'TickDir','out', ...
        'LineWidth',0.5, ...
        'Box','off', ...
        'XColor','k', ...
        'YColor','k');

    % ------------------ Significance markers ------------------------------
    yl = ylim(axL);
    yr = yl(2)-yl(1);

    % Three annotation rows:
    %   black = paired P vs S difference
    %   red   = P > 0
    %   blue  = S > 0
    yDiff_sig = yl(2)-0.040*yr;
    yP_sig    = yl(2)-0.105*yr;
    yS_sig    = yl(2)-0.170*yr;

    % P versus S: black
    if show_PvsS_significance && any(D.PvsS_sig_plot)

        if strcmpi(sig_correction_PvsS,'cluster') && ...
                strcmpi(cluster_plot_style,'bar')

            plot_sig_clusters( ...
                axL, ...
                time_points, ...
                D.PvsS_sig_plot, ...
                yDiff_sig, ...
                'k', ...
                1.3);

        else

            plot( ...
                axL, ...
                time_points(D.PvsS_sig_plot), ...
                repmat(yDiff_sig,1,nnz(D.PvsS_sig_plot)), ...
                'LineStyle','none', ...
                'Marker',PvsS_sig_marker, ...
                'Color','k', ...
                'MarkerSize',3.8, ...
                'HandleVisibility','off');
        end
    end

    % P > 0 / S > 0: colored
    if show_zero_significance

        if any(D.P_sig_plot)

            if strcmpi(sig_correction_zero,'cluster') && ...
                    strcmpi(cluster_plot_style,'bar')

                plot_sig_clusters( ...
                    axL, ...
                    time_points, ...
                    D.P_sig_plot, ...
                    yP_sig, ...
                    P_color, ...
                    1.3);

            else

                plot( ...
                    axL, ...
                    time_points(D.P_sig_plot), ...
                    repmat(yP_sig,1,nnz(D.P_sig_plot)), ...
                    'LineStyle','none', ...
                    'Marker',zero_sig_marker, ...
                    'Color',P_color, ...
                    'MarkerSize',3.5, ...
                    'HandleVisibility','off');
            end
        end

        if any(D.S_sig_plot)

            if strcmpi(sig_correction_zero,'cluster') && ...
                    strcmpi(cluster_plot_style,'bar')

                plot_sig_clusters( ...
                    axL, ...
                    time_points, ...
                    D.S_sig_plot, ...
                    yS_sig, ...
                    S_color, ...
                    1.3);

            else

                plot( ...
                    axL, ...
                    time_points(D.S_sig_plot), ...
                    repmat(yS_sig,1,nnz(D.S_sig_plot)), ...
                    'LineStyle','none', ...
                    'Marker',zero_sig_marker, ...
                    'Color',S_color, ...
                    'MarkerSize',3.5, ...
                    'HandleVisibility','off');
            end
        end
    end

    % if iclass==1
    %     legend(axL,[eP eS], ...
    %         {'Phonological','Semantic'}, ...
    %         'Box','off', ...
    %         'Location','best', ...
    %         'FontSize',6);
    % end

    hold(axL,'off');
end

title( ...
    tl, ...
    'Sequential representational dynamics in human sEEG contacts', ...
    'FontName','Arial', ...
    'FontSize',6, ...
    'FontWeight','bold');

% Small explanation of combined heatmap coding.
% annotation( ...
%     h1, ...
%     'textbox', ...
%     [0.05 0.002 0.90 0.025], ...
%     'String', ...
%     sprintf( ...
%     ['Combined heatmap: red = stronger positive phonological evidence; ' ...
%      'blue = stronger positive semantic evidence. ' ...
%      'Lines: participant-weighted mean \\pm SEM. ' ...
%      'Red/blue *: P/S > 0 (%s); black *: paired P vs S (%s); ' ...
%      '\\alpha = %.2f.'], ...
%      upper(sig_correction_zero), ...
%      upper(sig_correction_PvsS), ...
%      sig_alpha), ...
%     'EdgeColor','none', ...
%     'HorizontalAlignment','center', ...
%     'FontName','Arial', ...
%     'FontSize',6);

%% ======================= SUPPLEMENTARY HEATMAPS ==========================
%
% Same row ordering as the main combined heatmap.
% Left = P strength, right = S strength.

h2 = figure( ...
    'Color','w', ...
    'Position',[100 100 315 260], ...
    'Renderer','painters');

tl2 = tiledlayout( ...
    3,2, ...
    'TileSpacing','compact', ...
    'Padding','compact');

for iclass = 1:3

    D = ClassData(iclass);

    % ------------------------- P heatmap ---------------------------------
    axP = nexttile((iclass-1)*2+1);

    imagesc( ...
        axP, ...
        time_points, ...
        1:D.n_contacts, ...
        D.Hpho);

    colormap(axP,P_cmap);

    if separate_heatmap_positive_only
        caxis(axP,[0 heatmap_zmax]);
    else
        caxis(axP,[-heatmap_zmax heatmap_zmax]);
    end

    set(axP, ...
        'FontSize',6, ...
        'FontName','Arial', ...
        'TickDir','out', ...
        'TickLength',[0 0], ...
        'LineWidth',0.5, ...
        'Box','on');

    ylabel(axP,'Contact number');

    title(axP, ...
        sprintf('%s: phonological',D.title), ...
        'FontSize',6, ...
        'FontWeight','normal');

    if iclass==3
        xlabel(axP,'Time (ms)');
        xticks(axP,[0 200 400 600 800]);
    else
        xticks(axP,[0 200 400 600 800]);
        xticklabels(axP,[]);
    end

    % ------------------------- S heatmap ---------------------------------
    axS = nexttile((iclass-1)*2+2);

    imagesc( ...
        axS, ...
        time_points, ...
        1:D.n_contacts, ...
        D.Hsem);

    colormap(axS,S_cmap);

    if separate_heatmap_positive_only
        caxis(axS,[0 heatmap_zmax]);
    else
        caxis(axS,[-heatmap_zmax heatmap_zmax]);
    end

    set(axS, ...
        'FontSize',6, ...
        'FontName','Arial', ...
        'TickDir','out', ...
        'TickLength',[0 0], ...
        'LineWidth',0.5, ...
        'Box','on');

    ylabel(axS,'Contact number');

    title(axS, ...
        sprintf('%s: semantic',D.title), ...
        'FontSize',6, ...
        'FontWeight','normal');

    if iclass==3
        xlabel(axS,'Time (ms)');
        xticks(axS,[0 200 400 600 800]);
    else
        xticks(axS,[0 200 400 600 800]);
        xticklabels(axS,[]);
    end
end

title( ...
    tl2, ...
    'Separate phonological and semantic representation heatmaps', ...
    'FontName','Arial', ...
    'FontSize',6, ...
    'FontWeight','bold');

% annotation( ...
%     h2, ...
%     'textbox', ...
%     [0.05 0.002 0.90 0.025], ...
%     'String', ...
%     ['Rows use exactly the same contact order as the combined main heatmap. ' ...
%      'Heatmaps show positive permutation-normalized representation strength.'], ...
%     'EdgeColor','none', ...
%     'HorizontalAlignment','center', ...
%     'FontName','Arial', ...
%     'FontSize',6);

%% ====================== ADDITIONAL 0-600 ms VERSION ======================
%
% This section reproduces BOTH figures using only display points with
% window centers <= 600 ms. The underlying data, contact ordering, colors,
% participant weighting, SEM, and statistics are unchanged.

% ========================== MAIN FIGURE: 0-600 ms ==========================

h1_600 = figure( ...
    'Color','w', ...
    'Position',[100 100 315 260], ...
    'Renderer','painters');

tl_600 = tiledlayout( ...
    3,5, ...
    'TileSpacing','compact', ...
    'Padding','compact');

for iclass = 1:3

    D = ClassData(iclass);

    % --------------------- Combined heatmap -------------------------------
    tile_heat = (iclass-1)*5+1;

    axH = nexttile(tl_600,tile_heat,[1 2]);

    imagesc( ...
        axH, ...
        time_points_600, ...
        1:D.n_contacts, ...
        D.Hcombined(:,time_mask_600));

    colormap(axH,combined_cmap);

    caxis(axH,[-heatmap_zmax heatmap_zmax]);

    set(axH, ...
        'FontSize',6, ...
        'FontName','Arial', ...
        'TickDir','out', ...
        'TickLength',[0 0], ...
        'LineWidth',0.5, ...
        'Box','on', ...
        'XColor','k', ...
        'YColor','k');

    ylabel(axH,'Contact number','FontSize',6);

    title(axH, ...
        sprintf('%s sEEG contacts (n = %d)', ...
        D.title,D.n_contacts), ...
        'FontSize',6, ...
        'FontWeight','normal');

    % Keep the same heatmap x-axis behavior as the original 0-800 ms figure.
    % Do NOT force xlim to start at 0, because the first window center is 50 ms;
    % forcing [0 600] creates an artificial blank strip before the first column.
    axis(axH,'tight');

    if iclass==3
        xlabel(axH,'Time (ms)','FontSize',6);
        xticks(axH,[0 200 400 600]);
    else
        xticks(axH,[0 200 400 600]);
        xticklabels(axH,[]);
    end

    % --------------------- Line plot -------------------------------------
    tile_line = (iclass-1)*5+3;

    axL = nexttile(tl_600,tile_line,[1 3]);

    hold(axL,'on');

    if show_errorbars

        fill( ...
            axL, ...
            [time_points_600 fliplr(time_points_600)], ...
            [D.Pmean(time_mask_600)+D.Psem(time_mask_600) ...
             fliplr(D.Pmean(time_mask_600)-D.Psem(time_mask_600))], ...
            P_color, ...
            'FaceAlpha',0.18, ...
            'EdgeColor','none', ...
            'HandleVisibility','off');

        fill( ...
            axL, ...
            [time_points_600 fliplr(time_points_600)], ...
            [D.Smean(time_mask_600)+D.Ssem(time_mask_600) ...
             fliplr(D.Smean(time_mask_600)-D.Ssem(time_mask_600))], ...
            S_color, ...
            'FaceAlpha',0.18, ...
            'EdgeColor','none', ...
            'HandleVisibility','off');

        eP = plot( ...
            axL, ...
            time_points_600, ...
            D.Pmean(time_mask_600), ...
            '-o', ...
            'Color',P_color, ...
            'MarkerEdgeColor',P_color, ...
            'MarkerFaceColor','w', ...
            'LineWidth',1.0, ...
            'MarkerSize',3.2, ...
            'DisplayName','Phonological');

        eS = plot( ...
            axL, ...
            time_points_600, ...
            D.Smean(time_mask_600), ...
            '-o', ...
            'Color',S_color, ...
            'MarkerEdgeColor',S_color, ...
            'MarkerFaceColor','w', ...
            'LineWidth',1.0, ...
            'MarkerSize',3.2, ...
            'DisplayName','Semantic');

    else

        eP = plot( ...
            axL, ...
            time_points_600, ...
            D.Pmean(time_mask_600), ...
            '-o', ...
            'Color',P_color, ...
            'MarkerEdgeColor',P_color, ...
            'MarkerFaceColor','w', ...
            'LineWidth',1.0, ...
            'MarkerSize',3.2, ...
            'DisplayName','Phonological');

        eS = plot( ...
            axL, ...
            time_points_600, ...
            D.Smean(time_mask_600), ...
            '-o', ...
            'Color',S_color, ...
            'MarkerEdgeColor',S_color, ...
            'MarkerFaceColor','w', ...
            'LineWidth',1.0, ...
            'MarkerSize',3.2, ...
            'DisplayName','Semantic');
    end

    yline( ...
        axL,0,'--', ...
        'Color',[0.65 0.65 0.65], ...
        'LineWidth',0.5, ...
        'HandleVisibility','off');

    xlim(axL,[0 plot_end_600]);
    ylim(axL,class_line_ylim(iclass,:));

    xticks(axL,[0 200 400 600]);

    if iclass==3
        xlabel(axL,'Time (ms)','FontSize',6);
    else
        xticklabels(axL,[]);
    end

    ylabel(axL, ...
        {'Representation','index'}, ...
        'FontSize',6);

    title(axL, ...
        sprintf('%s | %d participants', ...
        D.title,D.n_subjects), ...
        'FontSize',6, ...
        'FontWeight','normal');

    set(axL, ...
        'FontSize',6, ...
        'FontName','Arial', ...
        'TickDir','out', ...
        'LineWidth',0.5, ...
        'Box','off', ...
        'XColor','k', ...
        'YColor','k');

    % ------------------ Significance markers ------------------------------
    yl = ylim(axL);
    yr = yl(2)-yl(1);

    yDiff_sig = yl(2)-0.040*yr;
    yP_sig    = yl(2)-0.105*yr;
    yS_sig    = yl(2)-0.170*yr;

    P_sig_600 = D.P_sig_plot & time_mask_600;
    S_sig_600 = D.S_sig_plot & time_mask_600;
    PvsS_sig_600 = D.PvsS_sig_plot & time_mask_600;

    % P versus S: black
    if show_PvsS_significance && any(PvsS_sig_600)

        if strcmpi(sig_correction_PvsS,'cluster') && ...
                strcmpi(cluster_plot_style,'bar')

            plot_sig_clusters( ...
                axL, ...
                time_points, ...
                PvsS_sig_600, ...
                yDiff_sig, ...
                'k', ...
                1.3);

        else

            plot( ...
                axL, ...
                time_points(PvsS_sig_600), ...
                repmat(yDiff_sig,1,nnz(PvsS_sig_600)), ...
                'LineStyle','none', ...
                'Marker',PvsS_sig_marker, ...
                'Color','k', ...
                'MarkerSize',3.8, ...
                'HandleVisibility','off');
        end
    end

    % P > 0 / S > 0: colored
    if show_zero_significance

        if any(P_sig_600)

            if strcmpi(sig_correction_zero,'cluster') && ...
                    strcmpi(cluster_plot_style,'bar')

                plot_sig_clusters( ...
                    axL, ...
                    time_points, ...
                    P_sig_600, ...
                    yP_sig, ...
                    P_color, ...
                    1.3);

            else

                plot( ...
                    axL, ...
                    time_points(P_sig_600), ...
                    repmat(yP_sig,1,nnz(P_sig_600)), ...
                    'LineStyle','none', ...
                    'Marker',zero_sig_marker, ...
                    'Color',P_color, ...
                    'MarkerSize',3.5, ...
                    'HandleVisibility','off');
            end
        end

        if any(S_sig_600)

            if strcmpi(sig_correction_zero,'cluster') && ...
                    strcmpi(cluster_plot_style,'bar')

                plot_sig_clusters( ...
                    axL, ...
                    time_points, ...
                    S_sig_600, ...
                    yS_sig, ...
                    S_color, ...
                    1.3);

            else

                plot( ...
                    axL, ...
                    time_points(S_sig_600), ...
                    repmat(yS_sig,1,nnz(S_sig_600)), ...
                    'LineStyle','none', ...
                    'Marker',zero_sig_marker, ...
                    'Color',S_color, ...
                    'MarkerSize',3.5, ...
                    'HandleVisibility','off');
            end
        end
    end

    % if iclass==1
    %     legend(axL,[eP eS], ...
    %         {'Phonological','Semantic'}, ...
    %         'Box','off', ...
    %         'Location','best', ...
    %         'FontSize',6);
    % end

    hold(axL,'off');
end

title( ...
    tl_600, ...
    'Sequential representational dynamics in human sEEG contacts', ...
    'FontName','Arial', ...
    'FontSize',6, ...
    'FontWeight','bold');

% annotation( ...
%     h1_600, ...
%     'textbox', ...
%     [0.05 0.002 0.90 0.025], ...
%     'String', ...
%     sprintf( ...
%     ['Combined heatmap: red = stronger positive phonological evidence; ' ...
%      'blue = stronger positive semantic evidence. ' ...
%      'Lines: participant-weighted mean \\pm SEM. ' ...
%      'Red/blue *: P/S > 0 (%s); black *: paired P vs S (%s); ' ...
%      '\\alpha = %.2f.'], ...
%      upper(sig_correction_zero), ...
%      upper(sig_correction_PvsS), ...
%      sig_alpha), ...
%     'EdgeColor','none', ...
%     'HorizontalAlignment','center', ...
%     'FontName','Arial', ...
%     'FontSize',5.5);


%% =========== ADDITIONAL HYBRID VERSION: FULL HEATMAP + 0-550 LINES ======
%
% Left panel keeps the FULL 0-800 ms heatmap.
% Right panel shows the participant-level trajectories only up to 550 ms.
% This is useful when the full temporal organization of contacts should stay
% visible, while the line plot is visually focused on the early sequential
% transition period.

h1_mix = figure( ...
    'Color','w', ...
    'Position',[100 100 315 260], ...
    'Renderer','painters');

tl_mix = tiledlayout( ...
    3,5, ...
    'TileSpacing','compact', ...
    'Padding','compact');

for iclass = 1:3

    D = ClassData(iclass);

    % --------------------- Combined heatmap (full 0-800 ms) --------------
    tile_heat = (iclass-1)*5+1;

    axH = nexttile(tl_mix,tile_heat,[1 2]);

    imagesc( ...
        axH, ...
        time_points, ...
        1:D.n_contacts, ...
        D.Hcombined);

    colormap(axH,combined_cmap);
    caxis(axH,[-heatmap_zmax heatmap_zmax]);

    set(axH, ...
        'FontSize',6, ...
        'FontName','Arial', ...
        'TickDir','out', ...
        'TickLength',[0 0], ...
        'LineWidth',0.5, ...
        'Box','on', ...
        'XColor','k', ...
        'YColor','k');

    ylabel(axH,'Contact number','FontSize',6);

    title(axH, ...
        sprintf('%s sEEG contacts (n = %d)', ...
        D.title,D.n_contacts), ...
        'FontSize',6, ...
        'FontWeight','normal');

    if iclass==3
        xlabel(axH,'Time (ms)','FontSize',6);
        xticks(axH,[0 200 400 600 800]);
    else
        xticks(axH,[0 200 400 600 800]);
        xticklabels(axH,[]);
    end

    % --------------------- Line plot (0-550 ms only) ---------------------
    tile_line = (iclass-1)*5+3;

    axL = nexttile(tl_mix,tile_line,[1 3]);

    hold(axL,'on');

    if show_errorbars

        fill( ...
            axL, ...
            [time_points_600 fliplr(time_points_600)], ...
            [D.Pmean(time_mask_600)+D.Psem(time_mask_600) ...
             fliplr(D.Pmean(time_mask_600)-D.Psem(time_mask_600))], ...
            P_color, ...
            'FaceAlpha',0.18, ...
            'EdgeColor','none', ...
            'HandleVisibility','off');

        fill( ...
            axL, ...
            [time_points_600 fliplr(time_points_600)], ...
            [D.Smean(time_mask_600)+D.Ssem(time_mask_600) ...
             fliplr(D.Smean(time_mask_600)-D.Ssem(time_mask_600))], ...
            S_color, ...
            'FaceAlpha',0.18, ...
            'EdgeColor','none', ...
            'HandleVisibility','off');

        plot( ...
            axL, ...
            time_points_600, ...
            D.Pmean(time_mask_600), ...
            '-o', ...
            'Color',P_color, ...
            'MarkerEdgeColor',P_color, ...
            'MarkerFaceColor','w', ...
            'LineWidth',1.0, ...
            'MarkerSize',2.2, ...
            'DisplayName','Phonological');

        plot( ...
            axL, ...
            time_points_600, ...
            D.Smean(time_mask_600), ...
            '-o', ...
            'Color',S_color, ...
            'MarkerEdgeColor',S_color, ...
            'MarkerFaceColor','w', ...
            'LineWidth',1.0, ...
            'MarkerSize',2.2, ...
            'DisplayName','Semantic');

    else

        plot( ...
            axL, ...
            time_points_600, ...
            D.Pmean(time_mask_600), ...
            '-o', ...
            'Color',P_color, ...
            'MarkerEdgeColor',P_color, ...
            'MarkerFaceColor','w', ...
            'LineWidth',1.0, ...
            'MarkerSize',2.2, ...
            'DisplayName','Phonological');

        plot( ...
            axL, ...
            time_points_600, ...
            D.Smean(time_mask_600), ...
            '-o', ...
            'Color',S_color, ...
            'MarkerEdgeColor',S_color, ...
            'MarkerFaceColor','w', ...
            'LineWidth',1.0, ...
            'MarkerSize',2.2, ...
            'DisplayName','Semantic');
    end

    yline( ...
        axL,0,'--', ...
        'Color',[0.65 0.65 0.65], ...
        'LineWidth',0.5, ...
        'HandleVisibility','off');

    xlim(axL,[0 plot_end_600]);
    ylim(axL,class_line_ylim(iclass,:));

    xticks(axL,[0 200 400 600]);

    if iclass==3
        xlabel(axL,'Time (ms)','FontSize',6);
    else
        xticklabels(axL,[]);
    end

    ylabel(axL, ...
        {'Representation','index'}, ...
        'FontSize',6);

    title(axL, ...
        sprintf('%s | %d participants', ...
        D.title,D.n_subjects), ...
        'FontSize',6, ...
        'FontWeight','normal');

    set(axL, ...
        'FontSize',6, ...
        'FontName','Arial', ...
        'TickDir','out', ...
        'LineWidth',0.5, ...
        'Box','off', ...
        'XColor','k', ...
        'YColor','k');

    % ------------------ Significance markers (0-550 ms) ------------------
    yl = ylim(axL);
    yr = yl(2)-yl(1);

    yDiff_sig = yl(2)-0.040*yr;
    yP_sig    = yl(2)-0.105*yr;
    yS_sig    = yl(2)-0.170*yr;

    P_sig_600 = D.P_sig_plot & time_mask_600;
    S_sig_600 = D.S_sig_plot & time_mask_600;
    PvsS_sig_600 = D.PvsS_sig_plot & time_mask_600;

    if show_PvsS_significance && any(PvsS_sig_600)

        if strcmpi(sig_correction_PvsS,'cluster') && ...
                strcmpi(cluster_plot_style,'bar')

            plot_sig_clusters( ...
                axL, ...
                time_points, ...
                PvsS_sig_600, ...
                yDiff_sig, ...
                'k', ...
                1.3);

        else

            plot( ...
                axL, ...
                time_points(PvsS_sig_600), ...
                repmat(yDiff_sig,1,nnz(PvsS_sig_600)), ...
                'LineStyle','none', ...
                'Marker',PvsS_sig_marker, ...
                'Color','k', ...
                'MarkerSize',3.8, ...
                'HandleVisibility','off');
        end
    end

    if show_zero_significance

        if any(P_sig_600)

            if strcmpi(sig_correction_zero,'cluster') && ...
                    strcmpi(cluster_plot_style,'bar')

                plot_sig_clusters( ...
                    axL, ...
                    time_points, ...
                    P_sig_600, ...
                    yP_sig, ...
                    P_color, ...
                    1.3);

            else

                plot( ...
                    axL, ...
                    time_points(P_sig_600), ...
                    repmat(yP_sig,1,nnz(P_sig_600)), ...
                    'LineStyle','none', ...
                    'Marker',zero_sig_marker, ...
                    'Color',P_color, ...
                    'MarkerSize',3.5, ...
                    'HandleVisibility','off');
            end
        end

        if any(S_sig_600)

            if strcmpi(sig_correction_zero,'cluster') && ...
                    strcmpi(cluster_plot_style,'bar')

                plot_sig_clusters( ...
                    axL, ...
                    time_points, ...
                    S_sig_600, ...
                    yS_sig, ...
                    S_color, ...
                    1.3);

            else

                plot( ...
                    axL, ...
                    time_points(S_sig_600), ...
                    repmat(yS_sig,1,nnz(S_sig_600)), ...
                    'LineStyle','none', ...
                    'Marker',zero_sig_marker, ...
                    'Color',S_color, ...
                    'MarkerSize',3.5, ...
                    'HandleVisibility','off');
            end
        end
    end

    hold(axL,'off');
end

title( ...
    tl_mix, ...
    'Sequential representational dynamics in human sEEG contacts', ...
    'FontName','Arial', ...
    'FontSize',6, ...
    'FontWeight','bold');

% annotation( ...
%     h1_mix, ...
%     'textbox', ...
%     [0.05 0.002 0.90 0.025], ...
%     'String', ...
%     sprintf( ...
%     ['Combined heatmap: full 0-800 ms. ' ...
%      'Lines: 0-%d ms participant-weighted mean \pm SEM. ' ...
%      'Red/blue *: P/S > 0 (%s); black *: paired P vs S (%s); ' ...
%      '\alpha = %.2f.'], ...
%      plot_end_600, ...
%      upper(sig_correction_zero), ...
%      upper(sig_correction_PvsS), ...
%      sig_alpha), ...
%     'EdgeColor','none', ...
%     'HorizontalAlignment','center', ...
%     'FontName','Arial', ...
%     'FontSize',5.5);


% ===================== SUPPLEMENTARY HEATMAPS: 0-600 ms ===================


h2_600 = figure( ...
    'Color','w', ...
    'Position',[100 100 315 260], ...
    'Renderer','painters');

tl2_600 = tiledlayout( ...
    3,2, ...
    'TileSpacing','compact', ...
    'Padding','compact');

for iclass = 1:3

    D = ClassData(iclass);

    % ------------------------- P heatmap ---------------------------------
    axP = nexttile(tl2_600,(iclass-1)*2+1);

    imagesc( ...
        axP, ...
        time_points_600, ...
        1:D.n_contacts, ...
        D.Hpho(:,time_mask_600));

    colormap(axP,P_cmap);

    if separate_heatmap_positive_only
        caxis(axP,[0 heatmap_zmax]);
    else
        caxis(axP,[-heatmap_zmax heatmap_zmax]);
    end

    set(axP, ...
        'FontSize',6, ...
        'FontName','Arial', ...
        'TickDir','out', ...
        'TickLength',[0 0], ...
        'LineWidth',0.5, ...
        'Box','on');

    ylabel(axP,'Contact number');

    title(axP, ...
        sprintf('%s: phonological',D.title), ...
        'FontSize',6, ...
        'FontWeight','normal');

    % Same convention as the original 0-800 ms heatmap.
    axis(axP,'tight');

    if iclass==3
        xlabel(axP,'Time (ms)');
        xticks(axP,[0 200 400 600]);
    else
        xticks(axP,[0 200 400 600]);
        xticklabels(axP,[]);
    end

    % ------------------------- S heatmap ---------------------------------
    axS = nexttile(tl2_600,(iclass-1)*2+2);

    imagesc( ...
        axS, ...
        time_points_600, ...
        1:D.n_contacts, ...
        D.Hsem(:,time_mask_600));

    colormap(axS,S_cmap);

    if separate_heatmap_positive_only
        caxis(axS,[0 heatmap_zmax]);
    else
        caxis(axS,[-heatmap_zmax heatmap_zmax]);
    end

    set(axS, ...
        'FontSize',6, ...
        'FontName','Arial', ...
        'TickDir','out', ...
        'TickLength',[0 0], ...
        'LineWidth',0.5, ...
        'Box','on');

    ylabel(axS,'Contact number');

    title(axS, ...
        sprintf('%s: semantic',D.title), ...
        'FontSize',6, ...
        'FontWeight','normal');

    % Same convention as the original 0-800 ms heatmap.
    axis(axS,'tight');

    if iclass==3
        xlabel(axS,'Time (ms)');
        xticks(axS,[0 200 400 600]);
    else
        xticks(axS,[0 200 400 600]);
        xticklabels(axS,[]);
    end
end

title( ...
    tl2_600, ...
    'Separate phonological and semantic representation heatmaps', ...
    'FontName','Arial', ...
    'FontSize',6, ...
    'FontWeight','bold');

% annotation( ...
%     h2_600, ...
%     'textbox', ...
%     [0.05 0.002 0.90 0.025], ...
%     'String', ...
%     ['Rows use exactly the same contact order as the combined main heatmap. ' ...
%      'Heatmaps show positive permutation-normalized representation strength.'], ...
%     'EdgeColor','none', ...
%     'HorizontalAlignment','center', ...
%     'FontName','Arial', ...
%     'FontSize',5.5);


%% =============================== SAVE ===================================

stats_csv = fullfile( ...
    save_dir, ...
    'Sequential_participant_level_statistics.csv');

writetable(stats_table,stats_csv);

main_base = fullfile( ...
    save_dir, ...
    'Sequential_main_combinedHeatmap_lines');

supp_base = fullfile( ...
    save_dir, ...
    'Sequential_supp_separate_P_S_heatmaps');

main_base_600 = fullfile( ...
    save_dir, ...
    'Sequential_main_combinedHeatmap_lines_0to600ms');

supp_base_600 = fullfile( ...
    save_dir, ...
    'Sequential_supp_separate_P_S_heatmaps_0to600ms');

main_base_mix = fullfile( ...
    save_dir, ...
    'Sequential_main_fullHeatmap_lines_0to550ms');

if save_png

    exportgraphics( ...
        h1, ...
        [main_base '.png'], ...
        'Resolution',png_resolution);

    exportgraphics( ...
        h2, ...
        [supp_base '.png'], ...
        'Resolution',png_resolution);

    exportgraphics( ...
        h1_600, ...
        [main_base_600 '.png'], ...
        'Resolution',png_resolution);

    exportgraphics( ...
        h2_600, ...
        [supp_base_600 '.png'], ...
        'Resolution',png_resolution);

    exportgraphics( ...
        h1_mix, ...
        [main_base_mix '.png'], ...
        'Resolution',png_resolution);
end

if save_pdf

    exportgraphics( ...
        h1, ...
        [main_base '.pdf'], ...
        'ContentType','vector');

    exportgraphics( ...
        h2, ...
        [supp_base '.pdf'], ...
        'ContentType','vector');

    exportgraphics( ...
        h1_600, ...
        [main_base_600 '.pdf'], ...
        'ContentType','vector');

    exportgraphics( ...
        h2_600, ...
        [supp_base_600 '.pdf'], ...
        'ContentType','vector');

    exportgraphics( ...
        h1_mix, ...
        [main_base_mix '.pdf'], ...
        'ContentType','vector');
end

fprintf('\n============================================================\n');
fprintf('SIGNIFICANCE DISPLAY\n');
fprintf('============================================================\n');
fprintf('Participant-first inference: contacts averaged within participant first.\n');
fprintf('Colored >0 markers shown: %s\n',string(show_zero_significance));
fprintf('Black P-vs-S markers shown: %s\n',string(show_PvsS_significance));
fprintf('P/S > 0 test: one-sided Wilcoxon signed-rank across participants.\n');
fprintf('P vs S test: paired two-sided Wilcoxon signed-rank across participants.\n');
fprintf('Zero-test correction: %s over 32 tests/class (16 P + 16 S).\n', ...
    upper(sig_correction_zero));
fprintf('P-vs-S correction: %s over 16 tests/class.\n', ...
    upper(sig_correction_PvsS));
fprintf('Corrected alpha: %.3f\n',sig_alpha);

if strcmpi(sig_correction_zero,'cluster') || ...
        strcmpi(sig_correction_PvsS,'cluster')

    fprintf('Cluster-forming alpha: %.3f\n',cluster_forming_alpha);
    fprintf('Cluster permutations: %d\n',cluster_nperm);
    fprintf('Cluster RNG seed: %d\n',cluster_seed);
    fprintf('Cluster plot style: %s\n',cluster_plot_style);
end

fprintf(['NOTE: full-data class-wise inference is descriptive/conditional because ' ...
    'the same dataset was used for unit phenotyping.\n']);

fprintf('\n============================================================\n');
fprintf('SEQUENTIAL PLOTTING COMPLETE\n');
fprintf('============================================================\n');
fprintf('Saved only the requested outputs to:\n%s\n',save_dir);
fprintf('  %s\n',stats_csv);

if save_png
    fprintf('  %s.png\n',main_base);
    fprintf('  %s.png\n',supp_base);
    fprintf('  %s.png\n',main_base_600);
    fprintf('  %s.png\n',supp_base_600);
end

if save_pdf
    fprintf('  %s.pdf\n',main_base);
    fprintf('  %s.pdf\n',supp_base);
    fprintf('  %s.pdf\n',main_base_600);
    fprintf('  %s.pdf\n',supp_base_600);
end

%% ============================== FUNCTIONS ===============================

function raw = load_strength_dataset(mat_file,preferred_names)
% Load the expected variable while remaining tolerant to minor naming changes.

S = load(mat_file);

names = fieldnames(S);

for i = 1:numel(preferred_names)

    if isfield(S,preferred_names{i})
        raw = S.(preferred_names{i});
        return;
    end
end

if numel(names)==1

    raw = S.(names{1});
    return;
end

error( ...
    ['Could not uniquely identify the strength variable in:\n%s\n' ...
     'Variables found: %s'], ...
    mat_file, ...
    strjoin(names,', '));

end


function T = standardize_strength_dataset(raw,nwin,label)
% Convert table/cell/mixed exported data into a standard MATLAB table:
%
% Subject | Contact | Psig | Ssig | UnitClass | T050 ... T800

expected_cols = 5+nwin;

if istable(raw)

    assert(width(raw)>=expected_cols, ...
        '%s dataset has only %d columns; expected >=%d.', ...
        label,width(raw),expected_cols);

    subject = string(raw{:,1});

    contact = numeric_column(raw{:,2});
    psig    = numeric_column(raw{:,3});
    ssig    = numeric_column(raw{:,4});
    unitcls = numeric_column(raw{:,5});

    strength = nan(height(raw),nwin);

    for w = 1:nwin
        strength(:,w) = numeric_column(raw{:,5+w});
    end

elseif iscell(raw)

    assert(size(raw,2)>=expected_cols, ...
        '%s dataset has only %d columns; expected >=%d.', ...
        label,size(raw,2),expected_cols);

    subject = string(raw(:,1));

    contact = numeric_column(raw(:,2));
    psig    = numeric_column(raw(:,3));
    ssig    = numeric_column(raw(:,4));
    unitcls = numeric_column(raw(:,5));

    strength = nan(size(raw,1),nwin);

    for w = 1:nwin
        strength(:,w) = numeric_column(raw(:,5+w));
    end

elseif isnumeric(raw)

    error( ...
        ['%s dataset is purely numeric, but column 1 is expected to contain ' ...
         'subject names. Please save/load the mixed metadata structure.'], ...
        label);

else

    error( ...
        'Unsupported %s dataset class: %s', ...
        label,class(raw));
end

T = table( ...
    subject(:), ...
    contact(:), ...
    psig(:), ...
    ssig(:), ...
    unitcls(:), ...
    'VariableNames',{ ...
    'Subject','Contact','Psig','Ssig','UnitClass'});

times = 50:50:(50*nwin);

for w = 1:nwin

    name = sprintf('T%03d',times(w));

    T.(name) = strength(:,w);
end

% Validation.
assert(all(ismember(unique(T.Psig),[0 1])), ...
    '%s Psig must contain only 0/1.',label);

assert(all(ismember(unique(T.Ssig),[0 1])), ...
    '%s Ssig must contain only 0/1.',label);

assert(all(ismember(unique(T.UnitClass),[0 1 2 3])), ...
    '%s UnitClass must contain only 0/1/2/3.',label);

assert(all(isfinite(T.Contact)), ...
    '%s Contact contains NaN/Inf.',label);

end


function x = numeric_column(v)
% Convert a table/cell/string/numeric column to double.

if isnumeric(v) || islogical(v)

    x = double(v);

elseif iscell(v)

    if isempty(v)
        x = [];
        return;
    end

    if all(cellfun(@(c) isnumeric(c) && isscalar(c),v))
        x = cellfun(@double,v);
    else
        x = str2double(string(v));
    end

elseif isstring(v) || ischar(v) || iscategorical(v)

    x = str2double(string(v));

else

    x = str2double(string(v));
end

x = x(:);

end


function p = one_sided_signrank_greater(x)
% Participant-level one-sided Wilcoxon signed-rank test:
% H1: median(x) > 0.
%
% Returns NaN when there are too few nonzero observations.

x = double(x(:));

x = x(isfinite(x));

if isempty(x)

    p = NaN;
    return;
end

% signrank drops exact zeros internally. Explicitly guard against all-zero.
if all(abs(x)<eps)

    p = 1;
    return;
end

try

    p = signrank( ...
        x, ...
        0, ...
        'tail','right');

catch

    % Compatibility fallback for MATLAB releases with different syntax.
    [p,~,stats] = signrank(x,0);

    if isfield(stats,'signedrank')

        % The two-sided fallback is conservatively converted only when
        % the sample median is positive.
        if median(x,'omitnan')>0
            p = p/2;
        else
            p = 1-p/2;
        end
    end
end

end


function [p,mean_delta] = paired_signrank_two_sided(x,y)
% Participant-level paired two-sided Wilcoxon signed-rank test for P vs S.
% Only participants with finite values for BOTH dimensions are retained.

x = double(x(:));
y = double(y(:));

valid = isfinite(x) & isfinite(y);

x = x(valid);
y = y(valid);

if isempty(x)

    p = NaN;
    mean_delta = NaN;
    return;
end

d = x-y;
mean_delta = mean(d,'omitnan');

if all(abs(d)<eps)

    p = 1;
    return;
end

try

    p = signrank(x,y,'tail','both');

catch

    % MATLAB's default signrank is two-sided in releases that do not
    % support the explicit 'tail','both' syntax.
    p = signrank(x,y);
end

end


function [P_corr,S_corr,P_sig,S_sig] = ...
    cluster_perm_joint_positive(Psub,Ssub,nperm,forming_alpha,cluster_alpha)
% Joint one-sided cluster permutation for P>0 and S>0.
%
% Participant-first data are nParticipant x nTime.
% The same participant sign flip is applied to the ENTIRE temporal
% trajectory and to BOTH P and S, preserving temporal dependence and
% within-participant dependence between representations.
%
% The permutation null stores the maximum positive cluster mass across
% BOTH P and S. This controls FWER jointly over representation dimension
% and time within the contact class.

Psub = double(Psub);
Ssub = double(Ssub);

nwin_local = size(Psub,2);

[tP,pP] = one_sample_t_and_p(Psub,'right');
[tS,pS] = one_sample_t_and_p(Ssub,'right');

maskP = isfinite(pP) & pP < forming_alpha & tP > 0;
maskS = isfinite(pS) & pS < forming_alpha & tS > 0;

[clustersP,massP] = positive_clusters(tP,maskP);
[clustersS,massS] = positive_clusters(tS,maskS);

null_max = zeros(nperm,1);

nsub = size(Psub,1);

for ip = 1:nperm

    flips = 2*(rand(nsub,1)>0.5)-1;

    Pperm = Psub .* flips;
    Sperm = Ssub .* flips;

    [tP_perm,pP_perm] = one_sample_t_and_p(Pperm,'right');
    [tS_perm,pS_perm] = one_sample_t_and_p(Sperm,'right');

    maskP_perm = isfinite(pP_perm) & ...
        pP_perm < forming_alpha & ...
        tP_perm > 0;

    maskS_perm = isfinite(pS_perm) & ...
        pS_perm < forming_alpha & ...
        tS_perm > 0;

    [~,massP_perm] = positive_clusters(tP_perm,maskP_perm);
    [~,massS_perm] = positive_clusters(tS_perm,maskS_perm);

    maxP = 0;
    maxS = 0;

    if ~isempty(massP_perm)
        maxP = max(massP_perm);
    end

    if ~isempty(massS_perm)
        maxS = max(massS_perm);
    end

    null_max(ip) = max(maxP,maxS);
end

P_corr = ones(1,nwin_local);
S_corr = ones(1,nwin_local);

for ic = 1:numel(clustersP)

    cluster_p = ...
        (1 + nnz(null_max >= massP(ic))) / ...
        (nperm + 1);

    P_corr(clustersP{ic}) = cluster_p;
end

for ic = 1:numel(clustersS)

    cluster_p = ...
        (1 + nnz(null_max >= massS(ic))) / ...
        (nperm + 1);

    S_corr(clustersS{ic}) = cluster_p;
end

P_sig = P_corr < cluster_alpha;
S_sig = S_corr < cluster_alpha;

end


function [corr_p,sig_mask] = ...
    cluster_perm_paired_twosided(Psub,Ssub,nperm,forming_alpha,cluster_alpha)
% Two-sided participant-level cluster permutation for paired P versus S.
%
% The tested trajectory is D = P-S.
% On every permutation, one sign is sampled per participant and applied to
% that participant's ENTIRE D trajectory.
%
% Positive and negative supra-threshold runs are treated as separate
% clusters. Cluster mass is sum(abs(t)) and the maximum cluster mass from
% each permutation forms the two-sided FWER null.

D = double(Psub) - double(Ssub);

nwin_local = size(D,2);

[tD,pD] = one_sample_t_and_p(D,'both');

mask = isfinite(pD) & pD < forming_alpha;

[clusters,masses] = signed_clusters_abs_mass(tD,mask);

null_max = zeros(nperm,1);

nsub = size(D,1);

for ip = 1:nperm

    flips = 2*(rand(nsub,1)>0.5)-1;

    Dperm = D .* flips;

    [t_perm,p_perm] = one_sample_t_and_p(Dperm,'both');

    mask_perm = isfinite(p_perm) & ...
        p_perm < forming_alpha;

    [~,mass_perm] = ...
        signed_clusters_abs_mass(t_perm,mask_perm);

    if isempty(mass_perm)
        null_max(ip) = 0;
    else
        null_max(ip) = max(mass_perm);
    end
end

corr_p = ones(1,nwin_local);

for ic = 1:numel(clusters)

    cluster_p = ...
        (1 + nnz(null_max >= masses(ic))) / ...
        (nperm + 1);

    corr_p(clusters{ic}) = cluster_p;
end

sig_mask = corr_p < cluster_alpha;

end


function [tval,pval] = one_sample_t_and_p(X,tail)
% Window-wise one-sample t statistic used only as the cluster-forming
% statistic. Cluster-level inference itself is nonparametric sign-flip.

X = double(X);

nwin_local = size(X,2);

tval = nan(1,nwin_local);
pval = nan(1,nwin_local);

for w = 1:nwin_local

    x = X(:,w);
    x = x(isfinite(x));

    n = numel(x);

    if n < 2
        continue;
    end

    mu = mean(x);
    sx = std(x,0);

    if sx <= eps

        if abs(mu) <= eps
            t = 0;
        elseif mu > 0
            t = Inf;
        else
            t = -Inf;
        end

    else

        t = mu / (sx/sqrt(n));
    end

    tval(w) = t;

    df = n-1;

    switch lower(tail)

        case 'right'
            pval(w) = 1-tcdf(t,df);

        case 'both'
            pval(w) = 2*(1-tcdf(abs(t),df));

        otherwise
            error('Unknown tail: %s',tail);
    end
end

end


function [clusters,masses] = positive_clusters(tval,mask)
% Contiguous positive supra-threshold clusters.

clusters = {};
masses = [];

runs = logical_runs(mask);

for ir = 1:size(runs,1)

    idx = runs(ir,1):runs(ir,2);

    % positive mask should already guarantee t>0
    idx = idx(tval(idx)>0);

    if isempty(idx)
        continue;
    end

    clusters{end+1} = idx; %#ok<AGROW>
    masses(end+1) = sum(tval(idx)); %#ok<AGROW>
end

end


function [clusters,masses] = signed_clusters_abs_mass(tval,mask)
% Contiguous two-sided clusters, separated by the sign of t.
%
% Adjacent positive and negative supra-threshold windows are NOT combined.

clusters = {};
masses = [];

positive_mask = mask & tval>0;
negative_mask = mask & tval<0;

runs_pos = logical_runs(positive_mask);
runs_neg = logical_runs(negative_mask);

for ir = 1:size(runs_pos,1)

    idx = runs_pos(ir,1):runs_pos(ir,2);

    clusters{end+1} = idx; %#ok<AGROW>
    masses(end+1) = sum(abs(tval(idx))); %#ok<AGROW>
end

for ir = 1:size(runs_neg,1)

    idx = runs_neg(ir,1):runs_neg(ir,2);

    clusters{end+1} = idx; %#ok<AGROW>
    masses(end+1) = sum(abs(tval(idx))); %#ok<AGROW>
end

end


function runs = logical_runs(mask)
% Return contiguous TRUE runs as [startIndex endIndex].

mask = logical(mask(:)');

d = diff([false mask false]);

starts = find(d==1);
stops = find(d==-1)-1;

runs = [starts(:) stops(:)];

end


function plot_sig_clusters(ax,time_points_local,sig_mask,y,color_value,line_width)
% Plot each contiguous significant cluster as a horizontal bar.
%
% The bar begins/ends at the centers of the first/last significant windows.
% This avoids implying temporal precision beyond the analyzed window centers.

runs = logical_runs(sig_mask);

for ir = 1:size(runs,1)

    idx = runs(ir,1):runs(ir,2);

    plot( ...
        ax, ...
        [time_points_local(idx(1)) time_points_local(idx(end))], ...
        [y y], ...
        '-', ...
        'Color',color_value, ...
        'LineWidth',line_width, ...
        'HandleVisibility','off');

    % If a significant cluster contains only one window, add a small dot so
    % it remains visible even though the line has zero horizontal length.
    if numel(idx)==1

        plot( ...
            ax, ...
            time_points_local(idx), ...
            y, ...
            'o', ...
            'Color',color_value, ...
            'MarkerFaceColor',color_value, ...
            'MarkerSize',2.5, ...
            'HandleVisibility','off');
    end
end

end


function p_adj = adjust_pvalues(p,method)
% Apply the requested multiple-comparison correction while preserving shape.

method = lower(string(method));

switch method

    case "holm"
        p_adj = holm_adjust(p);

    case "bonferroni"
        p_adj = bonferroni_adjust(p);

    case "fdr"
        p_adj = bh_adjust(p);

    case "none"
        p_adj = p;

    case "cluster"
        error(['Cluster permutation must be called on the participant x time ' ...
            'matrix before point-wise p-value adjustment.']);

    otherwise
        error('Unknown significance correction: %s',method);
end

end


function p_adj = holm_adjust(p)
% Holm-Bonferroni adjusted p-values (FWER control).

was_row = isrow(p);

p = double(p(:));
p_adj = nan(size(p));

valid = isfinite(p);
pv = p(valid);
m = numel(pv);

if m==0
    if was_row
        p_adj = p_adj';
    end
    return;
end

[p_sorted,ord] = sort(pv,'ascend');

adj_sorted = (m-(1:m)'+1) .* p_sorted;

% Step-down monotonicity.
for i = 2:m
    adj_sorted(i) = max(adj_sorted(i),adj_sorted(i-1));
end

adj_sorted = min(adj_sorted,1);

adj_valid = nan(m,1);
adj_valid(ord) = adj_sorted;

p_adj(valid) = adj_valid;

if was_row
    p_adj = p_adj';
end

end


function p_adj = bonferroni_adjust(p)
% Bonferroni adjusted p-values.

p_adj = double(p);
valid = isfinite(p_adj);
m = nnz(valid);

if m>0
    p_adj(valid) = min(p_adj(valid)*m,1);
end

end


function q = bh_adjust(p)
% Benjamini-Hochberg adjusted q-values.
% Input/output retain row orientation when possible.

was_row = isrow(p);

p = double(p(:));

q = nan(size(p));

valid = isfinite(p);

pv = p(valid);

m = numel(pv);

if m==0

    if was_row
        q = q';
    end

    return;
end

[p_sorted,ord] = sort(pv,'ascend');

q_sorted = p_sorted .* m ./ (1:m)';

for i = m-1:-1:1
    q_sorted(i) = min(q_sorted(i),q_sorted(i+1));
end

q_sorted = min(q_sorted,1);

qv = nan(m,1);
qv(ord) = q_sorted;

q(valid) = qv;

if was_row
    q = q';
end

end


function cmap = make_diverging_colormap(c1,c0,c2,n)
% Smooth c1 -> c0 -> c2 diverging colormap.

if mod(n,2)~=0
    n = n+1;
end

n1 = n/2;

left = [ ...
    linspace(c1(1),c0(1),n1)', ...
    linspace(c1(2),c0(2),n1)', ...
    linspace(c1(3),c0(3),n1)'];

right = [ ...
    linspace(c0(1),c2(1),n1)', ...
    linspace(c0(2),c2(2),n1)', ...
    linspace(c0(3),c2(3),n1)'];

cmap = [left;right];

end


function cmap = make_white_to_color_colormap(target,n)
% White -> target-color sequential colormap.

cmap = [ ...
    linspace(1,target(1),n)', ...
    linspace(1,target(2),n)', ...
    linspace(1,target(3),n)'];

end
