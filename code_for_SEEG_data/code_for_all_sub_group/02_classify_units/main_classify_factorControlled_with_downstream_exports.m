%% Main factor-controlled P / S / Shared classification + downstream exports
% =========================================================================
%
% PRIMARY CHANGE RELATIVE TO THE PREVIOUS SCRIPT
%
% Neural RDM:
%   The input remains the original 52 x 52 COSINE-DISTANCE neural RDM for
%   each responsive contact and each time window.
%
% Pairwise representation contrasts are changed to control the OTHER
% experimental factor:
%
%   Phonological-category effect:
%
%       E_P(t) =
%           mean[D(P-different, S-different)]
%           - mean[D(P-same, S-different)]
%
%       = mean of 650 different-P/different-S pairs
%         minus mean of the 26 true same-P/different-S pairs.
%
%   Semantic-category effect:
%
%       E_S(t) =
%           mean[D(P-different, S-different)]
%           - mean[D(P-different, S-same)]
%
%       = 650 versus 650 pairs.
%
% Thus:
%   - P is estimated while semantic-category relationship is held at
%     "different".
%   - S is estimated while phonological relationship is held at
%     "different".
%
% Significance:
%   Ordinary t-tests over RDM cells are NOT used.
%   Restricted stimulus-label permutation is used for every contact/window.
%
% Classification is intentionally kept UNCHANGED for this comparison:
%
%       nominal one-sided permutation p <= .05
%       observed effect > 0
%       >= 2 consecutive positive windows
%
%       P+ S- -> P-only
%       P- S+ -> S-only
%       P+ S+ -> Shared
%       P- S- -> None
%
% No P->S direction criterion is used for classification.
%
% Main plotted strength remains permutation-normalized:
%
%       Z = (observed effect - permutation-null mean) / null SD
%
% IMPORTANT:
%   This script does NOT regress out the acoustic RDM. It is intended to be
%   compared directly with the acoustic-controlled companion script.
%
% =========================================================================
%
% DOWNSTREAM EXPORTS ADDED IN THIS VERSION
% ----------------------------------------
% The primary classification is unchanged:
%   nominal one-sided restricted-permutation p <= .05,
%   observed effect > 0,
%   >= 2 consecutive positive windows.
%
% For the primary 200-ms / 50-ms scheme, this script additionally exports
% clean input variables for subsequent analyses of:
%
%   1) Sequential dynamics:
%      - all responsive contacts x 16 windows of P representation strength
%      - all responsive contacts x 16 windows of S representation strength
%
%   2) Representational geometry:
%      - all responsive contacts x 16 windows of 52x52 neural RDMs
%      - significant-window RDMs and strengths for P-only, S-only and Shared
%
%   3) Cortical hierarchy:
%      - subject/contact/electrode label/ROI + P/S/unit-class flags
%
% Export class code used downstream:
%      0 = non-selective / None
%      1 = P-only
%      2 = S-only
%      3 = Shared
%
% Representation strength exported below is the permutation-normalized Z:
%      Z = (observed effect - permutation-null mean) / permutation-null SD
%
% =========================================================================

clearvars;
close all;
clc;

%% ============================ USER SETTINGS =============================

num_permutations = 10000;
alpha = 0.05;
min_consecutive_windows = 2;

permutation_seed = 20260918;

% Primary analysis first:
%   1 = 200 ms / 50 ms
%   2 = 150 ms / 25 ms
%   [1 2] = both
scheme_indices = 1;  % fixed here because downstream tables require 16 windows

% Plot and save secondary raw-contrast figure.
save_raw_contrast_figure = true;

%% ================================ PATHS =================================

script_dir = fileparts(mfilename('fullpath'));

if isempty(script_dir)
    script_dir = pwd;
end

% Expected script location:
% ...\code_for_SEEG_data\code_for_all_sub_group\01_classify_units\
group_parent_dir = fileparts(script_dir);
root_dir = fileparts(group_parent_dir);

% Main output folder for the adopted primary classification.
save_dir = fullfile( ...
    script_dir, ...
    'main_factorControlled_2consecutive_with_downstream');

if ~isfolder(save_dir)
    mkdir(save_dir);
end

% Requested downstream-analysis inputs are placed one level lower and then
% separated by purpose.
downstream_dir = fullfile(save_dir,'downstream_analysis_inputs_200ms_50ms');
sequential_dir = fullfile(downstream_dir,'01_sequential_strength');
geometry_dir   = fullfile(downstream_dir,'02_geometry_RDM');
% Keep generated paths below MATLAB's practical Windows path limit.
sigwindow_dir  = fullfile(downstream_dir,'03_sigwin');
hierarchy_dir  = fullfile(downstream_dir,'04_hierarchy_ROI');

subdirs = {downstream_dir,sequential_dir,geometry_dir,sigwindow_dir,hierarchy_dir};
for id = 1:numel(subdirs)
    if ~isfolder(subdirs{id})
        mkdir(subdirs{id});
    end
end

% ROI table created previously from the current responsive-contact set.
roi_mat_file = fullfile( ...
    group_parent_dir, ...
    '01_arrange_ROI_MNI', ...
    'all_sub_resp_contacts_new.mat');

assert(isfile(roi_mat_file), ...
    'Cannot find ROI file:\n%s',roi_mat_file);

ROI = load(roi_mat_file,'all_sub_resp_contacts');

assert(isfield(ROI,'all_sub_resp_contacts'), ...
    'Variable all_sub_resp_contacts is missing from %s.',roi_mat_file);

all_sub_resp_contacts = ROI.all_sub_resp_contacts;

assert(iscell(all_sub_resp_contacts) && size(all_sub_resp_contacts,2)>=4, ...
    ['all_sub_resp_contacts must be an N x >=4 cell array: ' ...
     'subject, contact index, electrode label, ROI.']);

roi_subject = string(all_sub_resp_contacts(:,1));
roi_contact = nan(size(all_sub_resp_contacts,1),1);
roi_electrode_name = string(all_sub_resp_contacts(:,3));
roi_brain_region = string(all_sub_resp_contacts(:,4));

for ir = 1:size(all_sub_resp_contacts,1)
    roi_contact(ir) = parse_contact_index(all_sub_resp_contacts{ir,2});
end

roi_subject = upper(strtrim(roi_subject));

fprintf('\nLoaded ROI lookup: %d responsive-contact rows from:\n%s\n', ...
    size(all_sub_resp_contacts,1),roi_mat_file);

%% ============================= SUBJECTS =================================

subject_folder_names = { ...
    'sub01', ...
    'sub02', ...
    'sub03', ...
    'sub04', ...
    'sub05', ...
    'sub06', ...
    'sub07', ...
    'sub08', ...
    'sub09', ...
    'sub10', ...
    'sub11', ...
    'sub12', ...
    'sub13', ...
    'sub14', ...
    'sub15', ...
    'sub16', ...
    'sub17'};

subject_codes = { ...
    'sub01', ...
    'sub02', ...
    'sub03', ...
    'sub04', ...
    'sub05', ...
    'sub06', ...
    'sub07', ...
    'sub08', ...
    'sub09', ...
    'sub10', ...
    'sub11', ...
    'sub12', ...
    'sub13', ...
    'sub14', ...
    'sub15', ...
    'sub16', ...
    'sub17'};

num_subjects = numel(subject_codes);

%% ========================== WINDOW SCHEMES ==============================

schemes(1).tag = '200ms_50ms';
schemes(1).centers = 50:50:800;
schemes(1).nwin = 16;

schemes(2).tag = '150ms_25ms';
schemes(2).centers = 50:25:800;
schemes(2).nwin = 31;

assert(all(ismember(scheme_indices,1:numel(schemes))), ...
    'Invalid scheme_indices.');

%% =========================== PAIR INDEX =================================

num_conditions = 52;

lower_mask = tril(true(num_conditions),-1);
[pair_row,pair_col] = find(lower_mask);

num_pairs = numel(pair_row);

assert(num_pairs==1326, ...
    'Expected 1326 lower-triangle RDM cells.');

pair_index = zeros(num_conditions,num_conditions);

for k = 1:num_pairs
    pair_index(pair_row(k),pair_col(k)) = k;
end

%% ==================== RESTRICTED PERMUTATION BANKS ======================
%
% NEW FACTOR-CONTROLLED CONTRASTS
%
% The 52 words form three usable pair types:
%
%   1) P-same, S-different : 26
%      = the 26 true first-syllable+tone matched pairs
%
%   2) P-different, S-same : 650
%      = all within-category pairs
%
%   3) P-different, S-different : 650
%      = all cross-category pairs after excluding the 26 true P pairs
%
% Therefore:
%
%   P effect =
%       mean(P-different, S-different)
%       - mean(P-same, S-different)
%
%       = 650 pairs - 26 pairs
%
%   S effect =
%       mean(P-different, S-different)
%       - mean(P-different, S-same)
%
%       = 650 pairs - 650 pairs
%
% This controls the OTHER experimental factor by pair selection:
%   - P is tested only among S-different pairs.
%   - S is tested only among P-different pairs.
%
% IMPORTANT:
% RDM cells are still dependent, so ordinary pair-cell t-tests are NOT used.
% Significance is obtained from restricted stimulus-label permutations.
% -------------------------------------------------------------------------

fprintf('\n============================================================\n');
fprintf('Building FACTOR-CONTROLLED restricted permutation banks...\n');
fprintf('Permutations: %d\n',num_permutations);
fprintf('P contrast: Pdiff-Sdiff (650) - Psame-Sdiff (26)\n');
fprintf('S contrast: Pdiff-Sdiff (650) - Pdiff-Ssame (650)\n');
fprintf('============================================================\n');

rng(permutation_seed,'twister');

B = num_permutations;
Bplus = B+1;

% Fixed original semantic labels:
%   words 1:26  = category 1
%   words 27:52 = category 2
semantic_labels_true = uint8([ones(26,1); 2*ones(26,1)]);

cross_category_true = ...
    semantic_labels_true(pair_row) ~= semantic_labels_true(pair_col);

within_category_true = ...
    semantic_labels_true(pair_row) == semantic_labels_true(pair_col);

assert(nnz(cross_category_true)==676, ...
    'Expected 676 original cross-category pairs.');

assert(nnz(within_category_true)==650, ...
    'Expected 650 original within-category pairs.');

% Mask for the 26 TRUE phonological pairs.
true_P_pair_mask = false(num_pairs,1);

for i = 1:26
    idx = pair_index(i+26,i);
    assert(idx>0,'Invalid true phonological pair index.');
    true_P_pair_mask(idx) = true;
end

assert(nnz(true_P_pair_mask)==26, ...
    'Expected exactly 26 true P-same pairs.');

assert(all(cross_category_true(true_P_pair_mask)), ...
    'All true P-same pairs should be cross semantic-category.');

% =========================================================================
% P RESTRICTED NULL
% =========================================================================
%
% Observed:
%   P-same,S-different = the 26 true cross-category P pairs.
%   P-different,S-different = the remaining 650 cross-category pairs.
%
% Null:
%   Keep the two semantic groups FIXED.
%   Randomly create a new one-to-one matching between category 1 and
%   category 2. The 26 pseudo-matched pairs are treated as P-same;
%   the other 650 cross-category pairs are treated as P-different.
%
% Within-category pairs receive weight 0 and do not enter the P contrast.
% -------------------------------------------------------------------------

P_weights = zeros(num_pairs,Bplus,'single');

P_pairings = zeros(26,Bplus,'uint8');
P_pairings(:,1) = uint8(1:26);  % observed true pairing

for b = 2:Bplus
    P_pairings(:,b) = uint8(randperm(26));
end

for b = 1:Bplus

    % Start with all 676 S-different/cross-category pairs as candidate
    % P-different pairs.
    w = zeros(num_pairs,1,'single');
    w(cross_category_true) = single(1/650);

    pairing = double(P_pairings(:,b));

    % Overwrite the 26 selected one-to-one cross-category pairs as
    % P-same,S-different.
    for i = 1:26

        idx = pair_index(26+pairing(i),i);

        assert(idx>0, ...
            'Invalid phonological permutation pair index.');

        w(idx) = single(-1/26);
    end

    assert(nnz(w>0)==650, ...
        'P contrast must contain 650 Pdiff-Sdiff pairs.');

    assert(nnz(w<0)==26, ...
        'P contrast must contain 26 Psame-Sdiff pairs.');

    assert(nnz(w==0)==650, ...
        'P contrast must exclude the 650 within-category pairs.');

    P_weights(:,b) = w;
end

% =========================================================================
% S RESTRICTED NULL
% =========================================================================
%
% Observed:
%   P-different,S-same = all 650 within-category pairs.
%   P-different,S-different = 650 cross-category pairs after excluding the
%                            26 true P-same pairs.
%
% Null:
%   Preserve every TRUE P pair.
%   Randomly swap semantic category membership WITHIN each true P pair.
%   This guarantees:
%       - every true P pair remains cross-category;
%       - 26 words remain in each semantic group;
%       - the 26 true P pairs are always EXCLUDED from the S contrast.
%
% Thus every S permutation compares exactly:
%       650 Pdiff-Sdiff  versus  650 Pdiff-Ssame pairs.
% -------------------------------------------------------------------------

S_weights = zeros(num_pairs,Bplus,'single');

S_flips = false(26,Bplus);
S_flips(:,1) = false;  % observed true semantic assignment

for b = 2:Bplus
    S_flips(:,b) = rand(26,1)<0.5;
end

for b = 1:Bplus

    labels = zeros(num_conditions,1,'uint8');

    for i = 1:26

        if ~S_flips(i,b)

            labels(i)    = 1;
            labels(i+26) = 2;

        else

            labels(i)    = 2;
            labels(i+26) = 1;
        end
    end

    within = labels(pair_row)==labels(pair_col);
    between = ~within;

    % The true P pairs must remain between-category in every restricted
    % semantic permutation.
    assert(~any(within & true_P_pair_mask), ...
        'A true P pair became within-category; S permutation is invalid.');

    % Remove all 26 true P-same pairs from the S-different side.
    between_diffP = between & ~true_P_pair_mask;

    % All within-category pairs are automatically P-different under this
    % stimulus design, because every true P pair is cross-category.
    within_diffP = within;

    assert(nnz(between_diffP)==650, ...
        'Expected 650 Pdiff-Sdiff pairs in every S permutation.');

    assert(nnz(within_diffP)==650, ...
        'Expected 650 Pdiff-Ssame pairs in every S permutation.');

    w = zeros(num_pairs,1,'single');

    w(between_diffP) = single( 1/650);
    w(within_diffP)  = single(-1/650);

    % The 26 true P pairs do not contribute to S.
    assert(all(w(true_P_pair_mask)==0), ...
        'True P pairs must have zero weight in the S contrast.');

    S_weights(:,b) = w;
end

fprintf('Factor-controlled permutation banks ready.\n');

% Explicit observed-pair count diagnostics.
obsP_w = P_weights(:,1);
obsS_w = S_weights(:,1);

fprintf('\nObserved contrast pair counts:\n');
fprintf('  P: Pdiff-Sdiff = %d | Psame-Sdiff = %d | excluded = %d\n', ...
    nnz(obsP_w>0),nnz(obsP_w<0),nnz(obsP_w==0));
fprintf('  S: Pdiff-Sdiff = %d | Pdiff-Ssame = %d | excluded true-P = %d\n', ...
    nnz(obsS_w>0),nnz(obsS_w<0),nnz(obsS_w==0));


%% ============================== RESULTS =================================

group_table = table();
subject_table = table();
contact_table = table();
timecourse_table = table();

all_results = cell( ...
    num_subjects, ...
    numel(scheme_indices));

% =========================================================================
% DOWNSTREAM EXPORT VARIABLES
% =========================================================================
% These are populated only for the primary 200-ms / 50-ms scheme.
%
% All-responsive strength tables:
%   columns 1-5 = Subject, Contact, P_significant, S_significant, UnitClass
%   columns 6-21 = 16 window-wise permutation-normalized Z strengths.
allResponsive_P_strength = table();
allResponsive_S_strength = table();

% All-responsive RDM table:
%   columns 1-5 = same metadata
%   columns 6-21 = 16 cell-valued 52x52 neural RDMs.
allResponsive_RDM = table();

% Significant-window tables for the three selected unit classes.
% Each row is one contact x one significant window.
% The FIRST FOUR columns are exactly:
%   Subject | Contact | RDM | Strength
% Two explicit window-identification columns are appended to avoid losing the
% identity of the significant time window:
%   WindowIndex | WindowCenterMs
PhoUnits_P_sigWindows       = empty_sigwindow_table();
SemanticUnits_S_sigWindows  = empty_sigwindow_table();
SharedUnits_P_sigWindows    = empty_sigwindow_table();
SharedUnits_S_sigWindows    = empty_sigwindow_table();

% Hierarchy table:
% Subject | Contact | ElectrodeName | BrainRegion |
% P_significant | S_significant | UnitClass
allResponsive_ROI = table();

%% =============================== MAIN ===================================

for ii = 1:numel(scheme_indices)

    ischeme = scheme_indices(ii);

    tag = string(schemes(ischeme).tag);
    centers = double(schemes(ischeme).centers(:)');
    nwin = schemes(ischeme).nwin;

    fprintf('\n\n============================================================\n');
    fprintf('SCHEME: %s\n',tag);
    fprintf('============================================================\n');

    for isub = 1:num_subjects

        subject = subject_codes{isub};
        subject_folder = subject_folder_names{isub};

        fprintf('\n%s (%d/%d)\n', ...
            subject,isub,num_subjects);

        rdm_dir = fullfile( ...
            root_dir, ...
            ['code_for_' subject_folder], ...
            '03_construct_RDMs');

        rdm_file = fullfile( ...
            rdm_dir, ...
            sprintf( ...
            '%s_nchan_HGA_cosine_RDM_%s.mat', ...
            subject, ...
            char(tag)));

        assert(isfile(rdm_file), ...
            'Cannot find:\n%s', ...
            rdm_file);

        F = load(rdm_file,'nchan_rdm');

        nchan_rdm = F.nchan_rdm;

        assert(size(nchan_rdm,2)==nwin, ...
            '%s %s: expected %d windows, found %d.', ...
            subject,tag,nwin,size(nchan_rdm,2));

        nonempty = ~cellfun(@isempty,nchan_rdm);

        contacts = find(any(nonempty,2))';

        num_contacts = numel(contacts);
        num_rows = size(nchan_rdm,1);

        % -------------------------------------------------------------
        % Contact x time arrays.
        % -------------------------------------------------------------

        P_effect = nan(num_rows,nwin);
        S_effect = nan(num_rows,nwin);

        P_z = nan(num_rows,nwin);
        S_z = nan(num_rows,nwin);

        P_p = nan(num_rows,nwin);
        S_p = nan(num_rows,nwin);

        P_null_mean = nan(num_rows,nwin);
        S_null_mean = nan(num_rows,nwin);

        P_null_sd = nan(num_rows,nwin);
        S_null_sd = nan(num_rows,nwin);

        % -------------------------------------------------------------
        % Each contact: build one 1326 x time matrix, then evaluate all
        % windows against the same permutation banks in one matrix multiply.
        % -------------------------------------------------------------

        for ic = 1:num_contacts

            ch = contacts(ic);

            if mod(ic,10)==1 || ic==num_contacts
                fprintf('  contact %d/%d (channel %d)\n', ...
                    ic,num_contacts,ch);
            end

            Y = zeros(num_pairs,nwin,'single');

            for w = 1:nwin

                RDM = double(nchan_rdm{ch,w});

                assert(isequal(size(RDM),[52 52]), ...
                    '%s ch%d win%d RDM is not 52x52.', ...
                    subject,ch,w);

                y = RDM(lower_mask);

                assert(all(isfinite(y)), ...
                    ['%s ch%d win%d contains NaN/Inf. ' ...
                     'Off-diagonal zero distances are kept as valid.'], ...
                    subject,ch,w);

                Y(:,w) = single(y);
            end

            % ---------------------------------------------------------
            % Effects for observed + all permutations.
            %
            % poolP / poolS:
            %   rows = observed + permutations
            %   cols = time windows
            % ---------------------------------------------------------

            poolP = double((Y' * P_weights)');
            poolS = double((Y' * S_weights)');

            obsP = poolP(1,:);
            obsS = poolS(1,:);

            nullP = poolP(2:end,:);
            nullS = poolS(2:end,:);

            muP = mean(nullP,1);
            muS = mean(nullS,1);

            sdP = std(nullP,0,1);
            sdS = std(nullS,0,1);

            assert(all(isfinite(sdP) & sdP>eps), ...
                '%s ch%d: invalid P null SD.', ...
                subject,ch);

            assert(all(isfinite(sdS) & sdS>eps), ...
                '%s ch%d: invalid S null SD.', ...
                subject,ch);

            % ---------------------------------------------------------
            % One-sided nominal permutation p-values.
            % ---------------------------------------------------------

            pP = nan(1,nwin);
            pS = nan(1,nwin);

            for w = 1:nwin

                pP(w) = ...
                    (1 + nnz(nullP(:,w)>=obsP(w))) ...
                    / (B+1);

                pS(w) = ...
                    (1 + nnz(nullS(:,w)>=obsS(w))) ...
                    / (B+1);
            end

            % ---------------------------------------------------------
            % Permutation-normalized representation strength.
            % ---------------------------------------------------------

            zP = (obsP-muP)./sdP;
            zS = (obsS-muS)./sdS;

            % Store.
            P_effect(ch,:) = obsP;
            S_effect(ch,:) = obsS;

            P_z(ch,:) = zP;
            S_z(ch,:) = zS;

            P_p(ch,:) = pP;
            S_p(ch,:) = pS;

            P_null_mean(ch,:) = muP;
            S_null_mean(ch,:) = muS;

            P_null_sd(ch,:) = sdP;
            S_null_sd(ch,:) = sdS;
        end

        %% ==================== 2-CONSECUTIVE CLASSIFICATION ==============

        P_sig = ...
            P_p<=alpha & ...
            P_effect>0 & ...
            isfinite(P_p) & ...
            isfinite(P_effect);

        S_sig = ...
            S_p<=alpha & ...
            S_effect>0 & ...
            isfinite(S_p) & ...
            isfinite(S_effect);

        P_persistent = false(num_rows,1);
        S_persistent = false(num_rows,1);

        P_max_run = zeros(num_rows,1);
        S_max_run = zeros(num_rows,1);

        P_first_start = nan(num_rows,1);
        P_first_end = nan(num_rows,1);

        S_first_start = nan(num_rows,1);
        S_first_end = nan(num_rows,1);

        for ic = 1:num_contacts

            ch = contacts(ic);

            [ ...
                P_persistent(ch), ...
                P_max_run(ch), ...
                P_first_start(ch), ...
                P_first_end(ch)] = ...
                consecutive_info( ...
                    P_sig(ch,:), ...
                    min_consecutive_windows);

            [ ...
                S_persistent(ch), ...
                S_max_run(ch), ...
                S_first_start(ch), ...
                S_first_end(ch)] = ...
                consecutive_info( ...
                    S_sig(ch,:), ...
                    min_consecutive_windows);
        end

        % 1=P-only, 2=S-only, 3=Shared, 4=None.
        class_code = nan(num_rows,1);

        for ic = 1:num_contacts

            ch = contacts(ic);

            if P_persistent(ch) && ~S_persistent(ch)

                class_code(ch) = 1;

            elseif ~P_persistent(ch) && S_persistent(ch)

                class_code(ch) = 2;

            elseif P_persistent(ch) && S_persistent(ch)

                class_code(ch) = 3;

            else

                class_code(ch) = 4;
            end
        end

        cls = class_code(contacts);

        nP = nnz(cls==1);
        nS = nnz(cls==2);
        nShared = nnz(cls==3);
        nNone = nnz(cls==4);

        Ptotal = nP+nShared;
        Stotal = nS+nShared;

        fprintf('  P-only=%d | S-only=%d | Shared=%d | None=%d\n', ...
            nP,nS,nShared,nNone);

        %% ================= DOWNSTREAM EXPORTS: PRIMARY SCHEME ===========
        % Requested downstream variables are defined specifically for the
        % primary 16-window scheme (200-ms windows / 50-ms step).
        if ischeme==1

            assert(nwin==16, ...
                'Primary downstream export requires exactly 16 windows.');

            export_class = class_code(contacts);

            % Convert internal code 4=None into requested downstream code 0.
            export_class(export_class==4) = 0;

            P_flag = double(P_persistent(contacts));
            S_flag = double(S_persistent(contacts));

            n_export = numel(contacts);

            metaT = table( ...
                repmat(string(subject),n_export,1), ...
                contacts(:), ...
                P_flag(:), ...
                S_flag(:), ...
                export_class(:), ...
                'VariableNames',{ ...
                'Subject', ...
                'Contact', ...
                'P_significant', ...
                'S_significant', ...
                'UnitClass'});

            % -------------------------------------------------------------
            % 1) ALL RESPONSIVE CONTACTS: P/S REPRESENTATION STRENGTH
            % -------------------------------------------------------------
            strength_var_names = cellstr(compose('t%03dms',centers));

            TP = [ ...
                metaT, ...
                array2table( ...
                    P_z(contacts,:), ...
                    'VariableNames',strength_var_names)];

            TS = [ ...
                metaT, ...
                array2table( ...
                    S_z(contacts,:), ...
                    'VariableNames',strength_var_names)];

            allResponsive_P_strength = ...
                [allResponsive_P_strength;TP]; %#ok<AGROW>

            allResponsive_S_strength = ...
                [allResponsive_S_strength;TS]; %#ok<AGROW>

            % -------------------------------------------------------------
            % 2) ALL RESPONSIVE CONTACTS: 16 WINDOW-WISE NEURAL RDMs
            % -------------------------------------------------------------
            rdm_var_names = cellstr(compose('RDM_t%03dms',centers));

            RDM_cells = cell(n_export,nwin);

            for ie = 1:n_export

                ch_export = contacts(ie);

                for w = 1:nwin
                    % Preserve the original 52x52 cosine-distance RDM.
                    RDM_cells{ie,w} = nchan_rdm{ch_export,w};
                end
            end

            TRDM = [ ...
                metaT, ...
                cell2table( ...
                    RDM_cells, ...
                    'VariableNames',rdm_var_names)];

            allResponsive_RDM = ...
                [allResponsive_RDM;TRDM]; %#ok<AGROW>

            % -------------------------------------------------------------
            % 3) SELECTED UNIT CLASSES: SIGNIFICANT-WINDOW RDM + STRENGTH
            % -------------------------------------------------------------
            pho_contacts = ...
                contacts(class_code(contacts)==1);

            sem_contacts = ...
                contacts(class_code(contacts)==2);

            shared_contacts = ...
                contacts(class_code(contacts)==3);

            PhoUnits_P_sigWindows = [ ...
                PhoUnits_P_sigWindows; ...
                build_sigwindow_table( ...
                    subject, ...
                    pho_contacts, ...
                    P_sig, ...
                    P_z, ...
                    nchan_rdm, ...
                    centers)]; %#ok<AGROW>

            SemanticUnits_S_sigWindows = [ ...
                SemanticUnits_S_sigWindows; ...
                build_sigwindow_table( ...
                    subject, ...
                    sem_contacts, ...
                    S_sig, ...
                    S_z, ...
                    nchan_rdm, ...
                    centers)]; %#ok<AGROW>

            SharedUnits_P_sigWindows = [ ...
                SharedUnits_P_sigWindows; ...
                build_sigwindow_table( ...
                    subject, ...
                    shared_contacts, ...
                    P_sig, ...
                    P_z, ...
                    nchan_rdm, ...
                    centers)]; %#ok<AGROW>

            SharedUnits_S_sigWindows = [ ...
                SharedUnits_S_sigWindows; ...
                build_sigwindow_table( ...
                    subject, ...
                    shared_contacts, ...
                    S_sig, ...
                    S_z, ...
                    nchan_rdm, ...
                    centers)]; %#ok<AGROW>

            % -------------------------------------------------------------
            % 4) HIERARCHY: SUBJECT / CONTACT / ELECTRODE NAME / ROI
            % -------------------------------------------------------------
            electrode_names = strings(n_export,1);
            brain_regions = strings(n_export,1);

            subject_key = upper(strtrim(string(subject)));

            for ie = 1:n_export

                ch_export = contacts(ie);

                hit = find( ...
                    roi_subject==subject_key & ...
                    roi_contact==ch_export);

                if isempty(hit)
                    error([ ...
                        'ROI lookup failed for subject %s, contact %d. ' ...
                        'Check all_sub_resp_contacts_new.mat.'], ...
                        subject,ch_export);
                elseif numel(hit)>1
                    error([ ...
                        'ROI lookup is not unique for subject %s, contact %d.'], ...
                        subject,ch_export);
                end

                electrode_names(ie) = roi_electrode_name(hit);
                brain_regions(ie) = roi_brain_region(hit);
            end

            TROI = table( ...
                repmat(string(subject),n_export,1), ...
                contacts(:), ...
                electrode_names, ...
                brain_regions, ...
                P_flag(:), ...
                S_flag(:), ...
                export_class(:), ...
                'VariableNames',{ ...
                'Subject', ...
                'Contact', ...
                'ElectrodeName', ...
                'BrainRegion', ...
                'P_significant', ...
                'S_significant', ...
                'UnitClass'});

            allResponsive_ROI = ...
                [allResponsive_ROI;TROI]; %#ok<AGROW>
        end

        %% ======================= SUBJECT TABLE ===========================

        row = table( ...
            string(subject), ...
            tag, ...
            num_contacts, ...
            nP, ...
            nS, ...
            nShared, ...
            nNone, ...
            Ptotal, ...
            Stotal, ...
            'VariableNames',{ ...
            'subject', ...
            'window_scheme', ...
            'n_responsive_contacts', ...
            'phonological_only', ...
            'semantic_only', ...
            'shared', ...
            'none', ...
            'P_total', ...
            'S_total'});

        subject_table = [subject_table;row]; %#ok<AGROW>

        %% ======================= CONTACT TABLE ===========================

        class_names = { ...
            'Phonological only', ...
            'Semantic only', ...
            'Shared', ...
            'None'};

        for ic = 1:num_contacts

            ch = contacts(ic);

            P_start_ms = NaN;
            P_end_ms = NaN;
            S_start_ms = NaN;
            S_end_ms = NaN;

            if isfinite(P_first_start(ch))

                P_start_ms = ...
                    centers(P_first_start(ch));

                P_end_ms = ...
                    centers(P_first_end(ch));
            end

            if isfinite(S_first_start(ch))

                S_start_ms = ...
                    centers(S_first_start(ch));

                S_end_ms = ...
                    centers(S_first_end(ch));
            end

            row = table( ...
                string(subject), ...
                tag, ...
                ch, ...
                class_code(ch), ...
                string(class_names{class_code(ch)}), ...
                P_persistent(ch), ...
                S_persistent(ch), ...
                P_max_run(ch), ...
                S_max_run(ch), ...
                P_start_ms, ...
                P_end_ms, ...
                S_start_ms, ...
                S_end_ms, ...
                min(P_p(ch,:)), ...
                min(S_p(ch,:)), ...
                max(P_z(ch,:)), ...
                max(S_z(ch,:)), ...
                'VariableNames',{ ...
                'subject', ...
                'window_scheme', ...
                'contact', ...
                'class_code', ...
                'class_name', ...
                'P_persistent', ...
                'S_persistent', ...
                'P_max_consecutive_windows', ...
                'S_max_consecutive_windows', ...
                'P_first_run_start_center_ms', ...
                'P_first_run_end_center_ms', ...
                'S_first_run_start_center_ms', ...
                'S_first_run_end_center_ms', ...
                'P_min_nominal_p', ...
                'S_min_nominal_p', ...
                'P_peak_z', ...
                'S_peak_z'});

            contact_table = [contact_table;row]; %#ok<AGROW>
        end

        %% ================= SUBJECT-LEVEL CLASS CURVES ====================

        for class_id = 1:4

            class_contacts = ...
                contacts(class_code(contacts)==class_id);

            n_class_contacts = ...
                numel(class_contacts);

            if n_class_contacts>0

                Pz_curve = mean( ...
                    P_z(class_contacts,:), ...
                    1,'omitnan');

                Sz_curve = mean( ...
                    S_z(class_contacts,:), ...
                    1,'omitnan');

                Praw_curve = mean( ...
                    P_effect(class_contacts,:), ...
                    1,'omitnan');

                Sraw_curve = mean( ...
                    S_effect(class_contacts,:), ...
                    1,'omitnan');

            else

                Pz_curve = nan(1,nwin);
                Sz_curve = nan(1,nwin);

                Praw_curve = nan(1,nwin);
                Sraw_curve = nan(1,nwin);
            end

            for w = 1:nwin

                row = table( ...
                    string(subject), ...
                    tag, ...
                    class_id, ...
                    string(class_names{class_id}), ...
                    n_class_contacts, ...
                    centers(w), ...
                    Pz_curve(w), ...
                    Sz_curve(w), ...
                    Praw_curve(w), ...
                    Sraw_curve(w), ...
                    'VariableNames',{ ...
                    'subject', ...
                    'window_scheme', ...
                    'class_code', ...
                    'class_name', ...
                    'n_contacts_in_class', ...
                    'window_center_ms', ...
                    'phonological_z', ...
                    'semantic_z', ...
                    'phonological_raw_effect', ...
                    'semantic_raw_effect'});

                timecourse_table = ...
                    [timecourse_table;row]; %#ok<AGROW>
            end
        end

        %% ======================= STORE FULL RESULT =======================

        OUT = struct();

        OUT.subject = subject;
        OUT.scheme = char(tag);
        OUT.contacts = contacts;
        OUT.centers = centers;

        OUT.P_effect = P_effect;
        OUT.S_effect = S_effect;

        OUT.P_z = P_z;
        OUT.S_z = S_z;

        OUT.P_p = P_p;
        OUT.S_p = S_p;

        OUT.P_null_mean = P_null_mean;
        OUT.S_null_mean = S_null_mean;

        OUT.P_null_sd = P_null_sd;
        OUT.S_null_sd = S_null_sd;

        OUT.P_sig_window = P_sig;
        OUT.S_sig_window = S_sig;

        OUT.P_persistent = P_persistent;
        OUT.S_persistent = S_persistent;

        OUT.P_max_run = P_max_run;
        OUT.S_max_run = S_max_run;

        OUT.class_code = class_code;

        all_results{isub,ii} = OUT;
    end

    %% =========================== GROUP COUNTS ============================

    T = subject_table( ...
        subject_table.window_scheme==tag,:);

    N = sum(T.n_responsive_contacts);

    nP = sum(T.phonological_only);
    nS = sum(T.semantic_only);
    nShared = sum(T.shared);
    nNone = sum(T.none);

    row = table( ...
        tag, ...
        N, ...
        nP, ...
        nS, ...
        nShared, ...
        nNone, ...
        nP+nShared, ...
        nS+nShared, ...
        100*nP/N, ...
        100*nS/N, ...
        100*nShared/N, ...
        100*nNone/N, ...
        nnz(T.phonological_only>0), ...
        nnz(T.semantic_only>0), ...
        nnz(T.shared>0), ...
        'VariableNames',{ ...
        'window_scheme', ...
        'n_responsive_contacts', ...
        'phonological_only', ...
        'semantic_only', ...
        'shared', ...
        'none', ...
        'P_total', ...
        'S_total', ...
        'pct_phonological_only', ...
        'pct_semantic_only', ...
        'pct_shared', ...
        'pct_none', ...
        'subjects_with_phonological_only', ...
        'subjects_with_semantic_only', ...
        'subjects_with_shared'});

    group_table = [group_table;row]; %#ok<AGROW>

    %% ======================== MAIN Z-SCORE FIGURE ========================

    plot_four_class_timecourses( ...
        timecourse_table, ...
        subject_codes, ...
        tag, ...
        centers, ...
        'z', ...
        min_consecutive_windows, ...
        save_dir);

    %% ======================= SECONDARY RAW FIGURE ========================

    if save_raw_contrast_figure

        plot_four_class_timecourses( ...
            timecourse_table, ...
            subject_codes, ...
            tag, ...
            centers, ...
            'raw', ...
            min_consecutive_windows, ...
            save_dir);
    end
end

%% ==================== SAVE DOWNSTREAM ANALYSIS INPUTS ===================
%
% Sanity checks: the primary export should contain every responsive contact
% exactly once. The expected current total is determined from group_table,
% not hard-coded.

primary_tag = string(schemes(1).tag);
primary_group = group_table(group_table.window_scheme==primary_tag,:);

assert(height(primary_group)==1, ...
    'Expected exactly one group-summary row for the primary scheme.');

expected_N = primary_group.n_responsive_contacts;

assert(height(allResponsive_P_strength)==expected_N, ...
    'P-strength export row count does not match responsive-contact count.');

assert(height(allResponsive_S_strength)==expected_N, ...
    'S-strength export row count does not match responsive-contact count.');

assert(height(allResponsive_RDM)==expected_N, ...
    'RDM export row count does not match responsive-contact count.');

assert(height(allResponsive_ROI)==expected_N, ...
    'ROI export row count does not match responsive-contact count.');

% Check class coding requested by the downstream analyses.
assert(all(ismember(allResponsive_P_strength.UnitClass,[0 1 2 3])), ...
    'Unexpected downstream UnitClass code.');

% ---------------------- 01 sequential strength ---------------------------
P_strength_mat = fullfile( ...
    sequential_dir, ...
    'allResponsive_P_strength_16windows.mat');

S_strength_mat = fullfile( ...
    sequential_dir, ...
    'allResponsive_S_strength_16windows.mat');

P_strength_csv = fullfile( ...
    sequential_dir, ...
    'allResponsive_P_strength_16windows.csv');

S_strength_csv = fullfile( ...
    sequential_dir, ...
    'allResponsive_S_strength_16windows.csv');

save(P_strength_mat,'allResponsive_P_strength','-v7.3');
save(S_strength_mat,'allResponsive_S_strength','-v7.3');

writetable(allResponsive_P_strength,P_strength_csv);
writetable(allResponsive_S_strength,S_strength_csv);

% ------------------------ 02 geometry RDM -------------------------------
% Save contiguous row parts so each file remains small enough for a
% regular GitHub repository. The geometry plotting script joins them in
% order before analysis.
n_geometry_parts = 10;
n_geometry_rows = height(allResponsive_RDM);
for ipart = 1:n_geometry_parts
    first_row = floor((ipart-1) * n_geometry_rows / n_geometry_parts) + 1;
    last_row = floor(ipart * n_geometry_rows / n_geometry_parts);
    geometry_part.allResponsive_RDM = allResponsive_RDM(first_row:last_row,:);
    save(fullfile(geometry_dir, ...
        sprintf('allResponsive_RDM_part%02d.mat',ipart)), ...
        '-struct','geometry_part','-v7.3');
end
clear geometry_part;

% A lightweight metadata CSV is also saved because the matrix-valued RDM
% columns themselves cannot be represented faithfully in CSV.
allRDM_metadata = allResponsive_RDM(:,1:5);

writetable( ...
    allRDM_metadata, ...
    fullfile(geometry_dir,'allResponsive_RDM_metadata.csv'));

% ---------------- 03 selected significant-window inputs -----------------
sigwindow_mat = fullfile( ...
    sigwindow_dir, ...
    'selectedUnits_significantWindow_RDM_strength.mat');

save( ...
    sigwindow_mat, ...
    'PhoUnits_P_sigWindows', ...
    'SemanticUnits_S_sigWindows', ...
    'SharedUnits_P_sigWindows', ...
    'SharedUnits_S_sigWindows', ...
    '-v7.3');

% Save lightweight index/strength CSVs without the matrix-valued RDM column.
writetable( ...
    removevars(PhoUnits_P_sigWindows,'RDM'), ...
    fullfile(sigwindow_dir,'PhoUnits_P_sigWindows_index_strength.csv'));

writetable( ...
    removevars(SemanticUnits_S_sigWindows,'RDM'), ...
    fullfile(sigwindow_dir,'SemanticUnits_S_sigWindows_index_strength.csv'));

writetable( ...
    removevars(SharedUnits_P_sigWindows,'RDM'), ...
    fullfile(sigwindow_dir,'SharedUnits_P_sigWindows_index_strength.csv'));

writetable( ...
    removevars(SharedUnits_S_sigWindows,'RDM'), ...
    fullfile(sigwindow_dir,'SharedUnits_S_sigWindows_index_strength.csv'));

% ------------------------- 04 hierarchy ROI ------------------------------
hierarchy_mat = fullfile( ...
    hierarchy_dir, ...
    'allResponsive_ROI_unitClass.mat');

hierarchy_csv = fullfile( ...
    hierarchy_dir, ...
    'allResponsive_ROI_unitClass.csv');

save(hierarchy_mat,'allResponsive_ROI','-v7.3');
writetable(allResponsive_ROI,hierarchy_csv);

% ------------------------- variable dictionary ---------------------------
variable_name = [ ...
    "allResponsive_P_strength"; ...
    "allResponsive_S_strength"; ...
    "allResponsive_RDM"; ...
    "PhoUnits_P_sigWindows"; ...
    "SemanticUnits_S_sigWindows"; ...
    "SharedUnits_P_sigWindows"; ...
    "SharedUnits_S_sigWindows"; ...
    "allResponsive_ROI"];

description = [ ...
    "All responsive contacts; P Z-strength across 16 windows"; ...
    "All responsive contacts; S Z-strength across 16 windows"; ...
    "All responsive contacts; 52x52 cosine RDM across 16 windows"; ...
    "P-only contacts; P-significant contact x window RDM and Z-strength"; ...
    "S-only contacts; S-significant contact x window RDM and Z-strength"; ...
    "Shared contacts; P-significant contact x window RDM and Z-strength"; ...
    "Shared contacts; S-significant contact x window RDM and Z-strength"; ...
    "All responsive contacts; electrode label, ROI and unit class"];

variable_dictionary = table(variable_name,description);

writetable( ...
    variable_dictionary, ...
    fullfile(downstream_dir,'downstream_variable_dictionary.csv'));

fprintf('\n============================================================\n');
fprintf('DOWNSTREAM EXPORTS SAVED\n');
fprintf('============================================================\n');
fprintf('Root: %s\n',downstream_dir);
fprintf('All-responsive contacts exported: %d\n',expected_N);
fprintf('Pho significant-window rows: %d\n',height(PhoUnits_P_sigWindows));
fprintf('Semantic significant-window rows: %d\n',height(SemanticUnits_S_sigWindows));
fprintf('Shared-P significant-window rows: %d\n',height(SharedUnits_P_sigWindows));
fprintf('Shared-S significant-window rows: %d\n',height(SharedUnits_S_sigWindows));
fprintf('Hierarchy rows: %d\n',height(allResponsive_ROI));
fprintf('Blank ROI labels retained: %d\n',nnz(strlength(strtrim(allResponsive_ROI.BrainRegion))==0));

%% ================================ SAVE ==================================

group_csv = fullfile( ...
    save_dir, ...
    'factorControlled2consecutiveZ_group_counts.csv');

subject_csv = fullfile( ...
    save_dir, ...
    'factorControlled2consecutiveZ_subject_counts.csv');

contact_csv = fullfile( ...
    save_dir, ...
    'factorControlled2consecutiveZ_contact_results.csv');

timecourse_csv = fullfile( ...
    save_dir, ...
    'factorControlled2consecutiveZ_subject_class_timecourses.csv');

mat_file = fullfile( ...
    save_dir, ...
    'factorControlled2consecutiveZ_all_results.mat');

writetable(group_table,group_csv);
writetable(subject_table,subject_csv);
writetable(contact_table,contact_csv);
writetable(timecourse_table,timecourse_csv);

save( ...
    mat_file, ...
    'all_results', ...
    'group_table', ...
    'subject_table', ...
    'contact_table', ...
    'timecourse_table', ...
    'subject_codes', ...
    'subject_folder_names', ...
    'schemes', ...
    'scheme_indices', ...
    'num_permutations', ...
    'alpha', ...
    'min_consecutive_windows', ...
    'permutation_seed', ...
    'P_pairings', ...
    'S_flips', ...
    '-v7.3');

%% ============================== DISPLAY =================================

fprintf('\n\n============================================================\n');
fprintf('FINAL GROUP COUNTS\n');
fprintf('============================================================\n');

disp(group_table);

fprintf('\nOutputs:\n');
fprintf('%s\n',group_csv);
fprintf('%s\n',subject_csv);
fprintf('%s\n',contact_csv);
fprintf('%s\n',timecourse_csv);
fprintf('%s\n',mat_file);

fprintf('\nDownstream-analysis inputs:\n');
fprintf('%s\n',downstream_dir);

fprintf('\nMAIN FIGURE uses permutation-normalized Z scores.\n');
fprintf('NEW P contrast: Pdiff-Sdiff (650) - Psame-Sdiff (26).\n');
fprintf('NEW S contrast: Pdiff-Sdiff (650) - Pdiff-Ssame (650).\n');
fprintf(['Classification requires >=%d consecutive nominal p<=%.3f ' ...
         'positive-effect windows.\n'], ...
    min_consecutive_windows,alpha);

fprintf(['No multiple-comparison correction is applied across time ' ...
         'or contacts.\n']);

fprintf(['Class-specific trajectories use the same data as selection ' ...
         'and are descriptive.\n']);

%% ============================= FUNCTIONS ================================

function idx = parse_contact_index(x)
% Convert contact-index entries from numeric/string/cell content to double.

if isnumeric(x) || islogical(x)

    assert(isscalar(x) && isfinite(double(x)), ...
        'Invalid numeric contact index in ROI lookup.');

    idx = double(x);

elseif isstring(x) || ischar(x)

    idx = str2double(string(x));

    assert(isscalar(idx) && isfinite(idx), ...
        'Cannot parse contact index from ROI lookup: %s',string(x));

else

    error('Unsupported contact-index type in ROI lookup: %s',class(x));
end

assert(idx==fix(idx) && idx>=1, ...
    'Contact index must be a positive integer.');

end


function T = empty_sigwindow_table()
% Empty table with fixed variable types/order for selected-window exports.
%
% First four columns are exactly those requested by the downstream analyses.

T = table( ...
    strings(0,1), ...
    zeros(0,1), ...
    cell(0,1), ...
    zeros(0,1), ...
    zeros(0,1), ...
    zeros(0,1), ...
    'VariableNames',{ ...
    'Subject', ...
    'Contact', ...
    'RDM', ...
    'Strength', ...
    'WindowIndex', ...
    'WindowCenterMs'});

end


function T = build_sigwindow_table( ...
    subject, ...
    selected_contacts, ...
    sig_window_mask, ...
    z_strength, ...
    nchan_rdm, ...
    centers)
% Build one row per selected contact x nominally significant window.
%
% selected_contacts already belong to the requested unit class.
% sig_window_mask is the corresponding P_sig or S_sig matrix:
%       nominal restricted-permutation p<=.05 AND observed effect>0.
%
% The first four columns are:
%       Subject | Contact | RDM | Strength
% followed by explicit time-window identity.

T = empty_sigwindow_table();

for ic = 1:numel(selected_contacts)

    ch = selected_contacts(ic);

    sig_idx = find(sig_window_mask(ch,:));

    for k = 1:numel(sig_idx)

        w = sig_idx(k);

        one = table( ...
            string(subject), ...
            ch, ...
            {nchan_rdm{ch,w}}, ...
            z_strength(ch,w), ...
            w, ...
            centers(w), ...
            'VariableNames',{ ...
            'Subject', ...
            'Contact', ...
            'RDM', ...
            'Strength', ...
            'WindowIndex', ...
            'WindowCenterMs'});

        T = [T;one]; %#ok<AGROW>
    end
end

end


function [is_persistent,max_run,first_start,first_end] = ...
    consecutive_info(sig_vec,min_run)

sig_vec = logical(sig_vec(:)');

max_run = 0;
first_start = NaN;
first_end = NaN;

current_run = 0;
current_start = NaN;

for t = 1:numel(sig_vec)

    if sig_vec(t)

        if current_run==0
            current_start = t;
        end

        current_run = current_run+1;

        if current_run>max_run
            max_run = current_run;
        end

        if current_run>=min_run && isnan(first_start)

            first_start = current_start;
            first_end = t;
        end

    else

        current_run = 0;
        current_start = NaN;
    end
end

is_persistent = max_run>=min_run;

end


function plot_four_class_timecourses( ...
    timecourse_table, ...
    subject_codes, ...
    tag, ...
    centers, ...
    scale_type, ...
    min_consecutive_windows, ...
    save_dir)

num_subjects = numel(subject_codes);
nwin = numel(centers);

class_short = { ...
    'P-only', ...
    'S-only', ...
    'Shared', ...
    'None'};

if strcmpi(scale_type,'z')

    Pvar = 'phonological_z';
    Svar = 'semantic_z';

    ylabel_text = ...
        'Permutation-normalized representation strength (z)';

    title_scale = ...
        'Permutation-normalized representation strength';

    prefix = 'factorControlled2consecutiveZ';

elseif strcmpi(scale_type,'raw')

    Pvar = 'phonological_raw_effect';
    Svar = 'semantic_raw_effect';

    ylabel_text = ...
        'Raw neural-distance contrast';

    title_scale = ...
        'Raw representation contrast';

    prefix = 'nominal2consecutiveRAW';

else

    error('Unknown scale_type: %s',scale_type);
end

h = figure( ...
    'Color','w', ...
    'Position',[80 80 1200 850]);

tl = tiledlayout( ...
    2,2, ...
    'TileSpacing','compact', ...
    'Padding','compact');

title( ...
    tl, ...
    sprintf( ...
    ['%s | %s\n' ...
     'Classification: >=%d consecutive nominally significant windows'], ...
    title_scale, ...
    strrep(char(tag),'_','\_'), ...
    min_consecutive_windows), ...
    'FontWeight','bold');

all_plot_values = [];

for cls = 1:4

    nexttile;
    hold on;

    Psub = nan(num_subjects,nwin);
    Ssub = nan(num_subjects,nwin);

    nclass_contacts_total = 0;
    contributing_subjects = 0;

    for isub = 1:num_subjects

        subject = string(subject_codes{isub});

        use = ...
            timecourse_table.subject==subject & ...
            timecourse_table.window_scheme==tag & ...
            timecourse_table.class_code==cls;

        TT = timecourse_table(use,:);

        if isempty(TT)
            continue;
        end

        n_cls = unique(TT.n_contacts_in_class);

        if numel(n_cls)~=1
            error('Unexpected n_contacts_in_class duplication.');
        end

        if n_cls>0

            contributing_subjects = ...
                contributing_subjects+1;

            nclass_contacts_total = ...
                nclass_contacts_total+n_cls;

            for w = 1:nwin

                usew = ...
                    TT.window_center_ms==centers(w);

                Psub(isub,w) = ...
                    TT.(Pvar)(usew);

                Ssub(isub,w) = ...
                    TT.(Svar)(usew);
            end
        end
    end

    Pmean = mean(Psub,1,'omitnan');
    Smean = mean(Ssub,1,'omitnan');

    nP = sum(isfinite(Psub),1);
    nS = sum(isfinite(Ssub),1);

    Psem = std(Psub,0,1,'omitnan') ./ sqrt(nP);
    Ssem = std(Ssub,0,1,'omitnan') ./ sqrt(nS);

    xx = [centers fliplr(centers)];

    if any(isfinite(Pmean))

        fill( ...
            xx, ...
            [Pmean-Psem fliplr(Pmean+Psem)], ...
            [0.65 0.78 0.95], ...
            'EdgeColor','none', ...
            'FaceAlpha',0.28, ...
            'HandleVisibility','off');

        plot( ...
            centers, ...
            Pmean, ...
            '-', ...
            'Color',[0.05 0.30 0.80], ...
            'LineWidth',2.2, ...
            'DisplayName','Phonological');

        all_plot_values = ...
            [all_plot_values, ...
             Pmean-Psem, ...
             Pmean+Psem]; %#ok<AGROW>
    end

    if any(isfinite(Smean))

        fill( ...
            xx, ...
            [Smean-Ssem fliplr(Smean+Ssem)], ...
            [0.98 0.72 0.72], ...
            'EdgeColor','none', ...
            'FaceAlpha',0.28, ...
            'HandleVisibility','off');

        plot( ...
            centers, ...
            Smean, ...
            '-', ...
            'Color',[0.80 0.12 0.12], ...
            'LineWidth',2.2, ...
            'DisplayName','Semantic');

        all_plot_values = ...
            [all_plot_values, ...
             Smean-Ssem, ...
             Smean+Ssem]; %#ok<AGROW>
    end

    yline(0,'k--','HandleVisibility','off');

    xlabel('Window center (ms)');
    ylabel(ylabel_text);

    title(sprintf( ...
        '%s | %d contacts | %d subjects', ...
        class_short{cls}, ...
        nclass_contacts_total, ...
        contributing_subjects));

    grid on;
    box off;

    if cls==1
        legend('Location','best');
    end
end

% Same y-axis for all 4 panels.
finite_plot = ...
    all_plot_values(isfinite(all_plot_values));

if ~isempty(finite_plot)

    ymin = min(finite_plot);
    ymax = max(finite_plot);

    pad = 0.08*(ymax-ymin);

    if pad==0
        pad = max(abs([ymin ymax]))*0.1 + eps;
    end

    yl = [ymin-pad ymax+pad];

    ax = findall(h,'Type','axes');

    for ia = 1:numel(ax)
        ylim(ax(ia),yl);
    end
end

if strcmpi(scale_type,'z')

    note_text = ...
        ['Participant-weighted mean \pm SEM. ' ...
         'P and S are standardized relative to their own restricted ' ...
         'permutation null distributions. ' ...
         'Same data were used for classification and visualization; ' ...
         'curves are descriptive.'];

else

    note_text = ...
        ['Participant-weighted mean \pm SEM. ' ...
         'Raw effects are neural-distance contrasts. ' ...
         'Same data were used for classification and visualization; ' ...
         'curves are descriptive.'];
end

annotation( ...
    h, ...
    'textbox', ...
    [0.04 0.003 0.92 0.038], ...
    'String',note_text, ...
    'EdgeColor','none', ...
    'HorizontalAlignment','center', ...
    'FontSize',9);

png_file = fullfile( ...
    save_dir, ...
    sprintf( ...
    '%s_4class_timecourses_%s.png', ...
    prefix, ...
    char(tag)));

fig_file = fullfile( ...
    save_dir, ...
    sprintf( ...
    '%s_4class_timecourses_%s.fig', ...
    prefix, ...
    char(tag)));

exportgraphics( ...
    h, ...
    png_file, ...
    'Resolution',300);

savefig(h,fig_file);

end
