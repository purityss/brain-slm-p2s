%% step02_plot_hierarchy_12ROIs.m
% =========================================================================
% FINAL 12-ROI HIERARCHY ANALYSIS
%
% Primary hierarchy definition:
%   ROI order = median participant-level Euclidean distance to the
%   ipsilateral pmHG / TE1.1 reference.
%
% The ROI ordering is read from:
%   results_00_independent_ROI_hierarchy_rankings\
%       ROI_hierarchy_pmHG_distance_MEDIAN.csv
%
% Main analysis:
%   For each ROI:
%
%       P-only proportion
%         = N(P-only contacts) / N(all responsive contacts in ROI)
%
%       S-only proportion
%         = N(S-only contacts) / N(all responsive contacts in ROI)
%
%       Shared proportion
%         = N(Shared contacts) / N(all responsive contacts in ROI)
%
% Unit coding:
%   0 = None
%   1 = P-only sEEG contacts
%   2 = S-only sEEG contacts
%   3 = Shared sEEG contacts
%
% Formal hierarchy trend shown here:
%   ROI-level Spearman correlation between hierarchy rank and each
%   unit-class proportion.
%
% IMPORTANT:
%   - No smoothing is applied.
%   - No participant-level slope analysis is performed in this script.
%   - MFG and IFG are excluded because responsive coverage < 5 participants.
%   - Cuneus is excluded because it is a visual/occipital region.
%   - The final retained ROIs remain fine-grained and are NOT merged.
%
% OUTPUTS:
%   results_02_hierarchy\
%       Hierarchy_12ROI_distanceMedian_unit_proportions.png
%       Hierarchy_12ROI_distanceMedian_unit_proportions.pdf
%       Hierarchy_12ROI_distanceMedian_3x3_detail.png
%       Hierarchy_12ROI_distanceMedian_3x3_detail.pdf
%       Hierarchy_12ROI_distanceMedian_summary.csv
%       Hierarchy_12ROI_distanceMedian_Spearman.csv
%       Hierarchy_12ROI_distanceMedian_order.txt
%       Hierarchy_12ROI_distanceMedian_results.mat
%
% =========================================================================

clearvars;
close all;
clc;

%% ========================================================================
% 1. PATHS
% =========================================================================

script_dir = fileparts(mfilename('fullpath'));
input_root = fullfile(fileparts(script_dir), '02_classify_units', ...
    'main_factorControlled_2consecutive_with_downstream', ...
    'downstream_analysis_inputs_200ms_50ms');

plot_root = script_dir;

rank_root = fullfile( ...
    plot_root, ...
    'results_02_ROI_hierarchical_rankings');

save_dir = fullfile( ...
    plot_root, ...
    'results_02_hierarchical');

if ~isfolder(save_dir)
    mkdir(save_dir);
end

roi_file = fullfile( ...
    input_root, ...
    '04_hierarchy_ROI', ...
    'allResponsive_ROI_unitClass.mat');

rank_file = fullfile( ...
    rank_root, ...
    'ROI_hierarchy_pmHG_distance_MEDIAN.csv');

assert(isfile(roi_file), ...
    'Cannot find ROI file:\n%s',roi_file);

assert(isfile(rank_file), ...
    ['Cannot find distance-median ROI ranking file:\n%s\n' ...
     'Run step00_generate_ROI_hierarchy_rankings_mean_median.m first.'], ...
    rank_file);

%% ========================================================================
% 2. FINAL 12 ROIs
% =========================================================================
%
% This list defines WHICH ROIs are retained.
% Their DISPLAY ORDER will be taken entirely from the distance-median rank.

final_roi_set = [ ...
    "HG", ...
    "pmSTG", ...
    "Insula", ...
    "Sensorimotor", ...
    "aSTG", ...
    "MTG", ...
    "Temporal Pole", ...
    "ITG", ...
    "IPL", ...
    "PCC", ...
    "Precuneus", ...
    "MTL"];

nROI = numel(final_roi_set);

%% ========================================================================
% 3. COLORS / FIGURE SETTINGS
% =========================================================================

color_p      = [180 37 35]/255;
color_s      = [34 91 142]/255;
color_shared = [103 38 141]/255;

figure_position = [100 100 330 165];
png_resolution = 600;

y_lim = [0 80];
y_ticks = 0:20:80;

% 3 x 3 detailed figure
detail_figure_position = [100 100 900 650];

% Correlation panel:
%   points = RAW ROI proportions
%   fitted line = ordinary least-squares line for visualization only
%   annotation = Spearman rho and p from the formal hierarchy test
%
% Significance-star convention:
%   p < 0.001 : ***
%   p < 0.01  : **
%   p < 0.05  : *
%   otherwise : n.s.
%
% If p < 0.05, the correlation annotation is shown in red.

% -------------------------------------------------------------------------
% Display smoothing options
% -------------------------------------------------------------------------
% IMPORTANT:
% Smoothing is ONLY used for visualization.
% ROI proportions and Spearman statistics are always calculated from the
% original unsmoothed ROI values.
%
% true  = plot smoothed curves
% false = plot original raw ROI proportions
enable_smoothing = true;
% enable_smoothing = false;

% MATLAB smoothdata method, e.g.:
%   'movmean'   moving mean
%   'movmedian' moving median
%   'gaussian'  Gaussian-weighted moving average
smoothing_method = 'movmean';

% Number of adjacent ROIs used by smoothdata.
% Recommended values for 12 ROIs: 2, 3, or 4.
% A window of 3 is a mild smoothing.
smoothing_window = 2;

assert(isscalar(smoothing_window) && ...
       smoothing_window >= 1 && ...
       smoothing_window <= nROI && ...
       smoothing_window == round(smoothing_window), ...
       'smoothing_window must be an integer between 1 and %d.',nROI);

%% ========================================================================
% 4. LOAD CURRENT RESPONSIVE-CONTACT UNIT CLASSIFICATION
% =========================================================================

D = load(roi_file,'allResponsive_ROI');

assert(isfield(D,'allResponsive_ROI'), ...
    'ROI MAT does not contain allResponsive_ROI.');

ROI = D.allResponsive_ROI;

required_vars = { ...
    'Subject', ...
    'Contact', ...
    'ElectrodeName', ...
    'BrainRegion', ...
    'P_significant', ...
    'S_significant', ...
    'UnitClass'};

assert(all(ismember(required_vars,ROI.Properties.VariableNames)), ...
    'allResponsive_ROI is missing required columns.');

ROI.Subject = upper(strtrim(string(ROI.Subject)));
ROI.Contact = double(ROI.Contact);
ROI.ElectrodeName = string(ROI.ElectrodeName);
ROI.BrainRegion = string(ROI.BrainRegion);
ROI.UnitClass = double(ROI.UnitClass);

% Downstream coding:
% 0=None, 1=P-only, 2=S-only, 3=Shared
assert(all(ismember(unique(ROI.UnitClass),[0 1 2 3])), ...
    'Unexpected UnitClass coding.');

fprintf('\n============================================================\n');
fprintf('CURRENT RESPONSIVE CONTACT SET\n');
fprintf('============================================================\n');
fprintf('All responsive contacts = %d\n',height(ROI));

%% ========================================================================
% 5. MAP DETAILED ANATOMICAL LABELS TO FINAL 12 ROIs
% =========================================================================

ROI.ROI12 = strings(height(ROI),1);

for i = 1:height(ROI)
    ROI.ROI12(i) = map_brain_region_to_roi12(ROI.BrainRegion(i));
end

keep = ismember(ROI.ROI12,final_roi_set);

H = ROI(keep,:);

fprintf('Responsive contacts in final 12 ROIs = %d\n',height(H));
fprintf('Participants represented = %d\n',numel(unique(H.Subject)));

%% ========================================================================
% 6. READ DISTANCE-MEDIAN ROI ORDER
% =========================================================================

R = readtable( ...
    rank_file, ...
    'VariableNamingRule','preserve');

assert(ismember('ROI',R.Properties.VariableNames), ...
    'Ranking CSV lacks ROI column.');

assert(ismember('Rank_distance_median',R.Properties.VariableNames), ...
    'Ranking CSV lacks Rank_distance_median column.');

assert(ismember('Median_of_subject_medians',R.Properties.VariableNames), ...
    'Ranking CSV lacks Median_of_subject_medians column.');

R.ROI = string(R.ROI);

% Keep exactly the final 12 ROIs
R = R(ismember(R.ROI,final_roi_set),:);

assert(height(R)==nROI, ...
    'Distance-median rank file does not contain exactly the final 12 ROIs.');

assert(all(ismember(final_roi_set,R.ROI)), ...
    'At least one final ROI is missing from the distance-median rank file.');

R = sortrows(R,'Rank_distance_median','ascend');

roi_order = R.ROI;
roi_distance_mm = R.Median_of_subject_medians;

fprintf('\n============================================================\n');
fprintf('FINAL ROI ORDER: MEDIAN DISTANCE TO pmHG\n');
fprintf('============================================================\n');

for ir = 1:nROI
    fprintf('%2d. %-15s  %.3f mm\n', ...
        ir, ...
        char(roi_order(ir)), ...
        roi_distance_mm(ir));
end

%% ========================================================================
% 7. CALCULATE ROI UNIT PROPORTIONS
% =========================================================================
%
% Denominator is ALL responsive contacts within that ROI.

summary_table = table();

for ir = 1:nROI

    roi = roi_order(ir);

    use = H.ROI12==roi;
    cls = H.UnitClass(use);

    n_resp   = nnz(use);
    n_p      = nnz(cls==1);
    n_s      = nnz(cls==2);
    n_shared = nnz(cls==3);
    n_none   = nnz(cls==0);

    if n_resp>0

        pct_p      = 100*n_p/n_resp;
        pct_s      = 100*n_s/n_resp;
        pct_shared = 100*n_shared/n_resp;

    else

        pct_p = NaN;
        pct_s = NaN;
        pct_shared = NaN;
    end

    row = table( ...
        ir, ...
        roi, ...
        roi_distance_mm(ir), ...
        n_resp, ...
        n_p, ...
        n_s, ...
        n_shared, ...
        n_none, ...
        pct_p, ...
        pct_s, ...
        pct_shared, ...
        'VariableNames',{ ...
        'HierarchyRank', ...
        'ROI', ...
        'MedianDistanceToPmHG_mm', ...
        'N_responsive', ...
        'N_Ponly', ...
        'N_Sonly', ...
        'N_Shared', ...
        'N_None', ...
        'Pct_Ponly', ...
        'Pct_Sonly', ...
        'Pct_Shared'});

    summary_table = [summary_table;row]; %#ok<AGROW>
end

%% ========================================================================
% 8. ROI-LEVEL SPEARMAN HIERARCHY TEST
% =========================================================================

rank_x = summary_table.HierarchyRank;

[rho_P,p_P] = corr( ...
    rank_x, ...
    summary_table.Pct_Ponly, ...
    'Type','Spearman', ...
    'Rows','complete');

[rho_S,p_S] = corr( ...
    rank_x, ...
    summary_table.Pct_Sonly, ...
    'Type','Spearman', ...
    'Rows','complete');

[rho_Shared,p_Shared] = corr( ...
    rank_x, ...
    summary_table.Pct_Shared, ...
    'Type','Spearman', ...
    'Rows','complete');

spearman_table = table( ...
    ["P-only";"S-only";"Shared"], ...
    [rho_P;rho_S;rho_Shared], ...
    [p_P;p_S;p_Shared], ...
    'VariableNames',{ ...
    'UnitClass', ...
    'SpearmanRho', ...
    'Pvalue'});

fprintf('\n============================================================\n');
fprintf('ROI-LEVEL SPEARMAN HIERARCHY RESULTS\n');
fprintf('============================================================\n');
disp(spearman_table);

%% ========================================================================
% 9. PREPARE RAW AND SMOOTH DISPLAY VALUES
% =========================================================================
%
% These variables affect ONLY the plotted curves.
% The summary table and all statistical tests above remain UNSMOOTHED.

plot_P_raw = summary_table.Pct_Ponly;
plot_S_raw = summary_table.Pct_Sonly;
plot_Shared_raw = summary_table.Pct_Shared;

plot_P_smooth = plot_P_raw;
plot_S_smooth = plot_S_raw;
plot_Shared_smooth = plot_Shared_raw;

if enable_smoothing

    plot_P_smooth = smoothdata( ...
        plot_P_raw, ...
        'movmean', ...
        smoothing_window);

    plot_S_smooth = smoothdata( ...
        plot_S_raw, ...
        'movmean', ...
        smoothing_window);

    plot_Shared_smooth = smoothdata( ...
        plot_Shared_raw, ...
        'movmean', ...
        smoothing_window);

    % If another supported method is requested, apply it instead.
    if ~strcmpi(smoothing_method,'movmean')

        plot_P_smooth = smoothdata( ...
            plot_P_raw, ...
            smoothing_method, ...
            smoothing_window);

        plot_S_smooth = smoothdata( ...
            plot_S_raw, ...
            smoothing_method, ...
            smoothing_window);

        plot_Shared_smooth = smoothdata( ...
            plot_Shared_raw, ...
            smoothing_method, ...
            smoothing_window);
    end
end

% Save display values explicitly so the plotted curves can be reproduced.
summary_table.Display_Pct_Ponly_raw = plot_P_raw;
summary_table.Display_Pct_Sonly_raw = plot_S_raw;
summary_table.Display_Pct_Shared_raw = plot_Shared_raw;
summary_table.Display_Pct_Ponly_smooth = plot_P_smooth;
summary_table.Display_Pct_Sonly_smooth = plot_S_smooth;
summary_table.Display_Pct_Shared_smooth = plot_Shared_smooth;

%% ========================================================================
% 10. MAIN FIGURES (RAW + SMOOTH)
% =========================================================================

curve_modes = {'raw','smooth'};
main_fig_handles = cell(1,2);
detail_fig_handles = cell(1,2);

for imode = 1:2

    this_mode = curve_modes{imode};

    if strcmpi(this_mode,'raw')
        plot_P = plot_P_raw;
        plot_S = plot_S_raw;
        plot_Shared = plot_Shared_raw;
        mode_label = 'raw';
    else
        plot_P = plot_P_smooth;
        plot_S = plot_S_smooth;
        plot_Shared = plot_Shared_smooth;
        mode_label = sprintf('smooth_%s_w%d',lower(smoothing_method),smoothing_window);
    end

    %% MAIN FIGURE
    h = figure( ...
        'Color','w', ...
        'Position',figure_position, ...
        'Renderer','painters');

    ax = axes( ...
        h, ...
        'Position',[0.15 0.24 0.82 0.61]);

    hold(ax,'on');

    x = 1:nROI;

    plot(ax,x,plot_P,'-','Color',color_p,'LineWidth',1.5);
    plot(ax,x,plot_Shared,'-','Color',color_shared,'LineWidth',1.5);
    plot(ax,x,plot_S,'-','Color',color_s,'LineWidth',1.5);

    xlim(ax,[0.5 nROI+0.5]);
    ylim(ax,y_lim);
    xticks(ax,x);
    xticklabels(ax,summary_table.ROI);
    xtickangle(ax,90);
    yticks(ax,y_ticks);

    xlabel(ax,'Brain area','FontName','Arial','FontSize',6);
    ylabel(ax,'Proportion of sEEG contacts (%)','FontName','Arial','FontSize',6);

    set(ax,'FontName','Arial','FontSize',6,'Box','off','TickDir','out','LineWidth',0.5,'XColor','k','YColor','k');

    % annotation(h,'rectangle',[0.12 0.885 0.85 0.085],'FaceColor',[0.95 0.95 0.95],'EdgeColor','none');
    % annotation(h,'textbox',[0.12 0.888 0.85 0.075],'String',sprintf('Human Brain (%s)',strrep(mode_label,'_',' ')),'EdgeColor','none','HorizontalAlignment','center','VerticalAlignment','middle','FontName','Arial','FontSize',6,'FontWeight','bold');
    hold(ax,'off');
    main_fig_handles{imode} = h;

    %% DETAILED 3x3 FIGURE
    h_detail = figure('Color','w','Position',detail_figure_position,'Renderer','painters');
    tl_detail = tiledlayout(h_detail,3,3,'TileSpacing','compact','Padding','compact');

    unit_names_detail = ["P-only sEEG contacts","S-only sEEG contacts","Shared sEEG contacts"];
    unit_colors_detail = {color_p,color_s,color_shared};

    raw_prop_detail = {summary_table.Pct_Ponly,summary_table.Pct_Sonly,summary_table.Pct_Shared};
    display_prop_detail = {plot_P,plot_S,plot_Shared};
    rho_detail = [rho_P,rho_S,rho_Shared];
    p_detail = [p_P,p_S,p_Shared];

    for iu = 1:3
        this_color = unit_colors_detail{iu};
        y_raw = raw_prop_detail{iu};
        y_display = display_prop_detail{iu};

        ax_d = nexttile(tl_detail,(iu-1)*3+1);
        plot(ax_d,x,summary_table.MedianDistanceToPmHG_mm,'-o','Color',[0.25 0.25 0.25],'MarkerEdgeColor',[0.25 0.25 0.25],'MarkerFaceColor','w','MarkerSize',3.8,'LineWidth',1.4);
        xlim(ax_d,[0.5 nROI+0.5]);
        xticks(ax_d,x); xticklabels(ax_d,summary_table.ROI); xtickangle(ax_d,90);
        ylabel(ax_d,sprintf('%s\nDistance (mm)',unit_names_detail(iu)),'FontName','Arial','FontSize',6);
        if iu==1
            title(ax_d,'Distance hierarchy','FontName','Arial','FontSize',6,'FontWeight','normal');
        end
        set(ax_d,'FontName','Arial','FontSize',6,'Box','off','TickDir','out','LineWidth',0.5,'XColor','k','YColor','k');

        ax_p = nexttile(tl_detail,(iu-1)*3+2);
        plot(ax_p,x,y_display,'-o','Color',this_color,'MarkerEdgeColor',this_color,'MarkerFaceColor','w','MarkerSize',3.8,'LineWidth',1.6);
        xlim(ax_p,[0.5 nROI+0.5]); ylim(ax_p,y_lim);
        xticks(ax_p,x); xticklabels(ax_p,summary_table.ROI); xtickangle(ax_p,90); yticks(ax_p,y_ticks);
        ylabel(ax_p,'Proportion (%)','FontName','Arial','FontSize',6);
        if iu==1
            if strcmpi(this_mode,'raw')
                title(ax_p,'sEEG contact proportion (raw)','FontName','Arial','FontSize',6,'FontWeight','normal');
            else
                title(ax_p,sprintf('sEEG contact proportion (%s, w=%d)',smoothing_method,smoothing_window),'FontName','Arial','FontSize',6,'FontWeight','normal');
            end
        end
        set(ax_p,'FontName','Arial','FontSize',6,'Box','off','TickDir','out','LineWidth',0.5,'XColor','k','YColor','k');

        ax_c = nexttile(tl_detail,(iu-1)*3+3);
        x_corr = summary_table.MedianDistanceToPmHG_mm;
        y_corr = y_raw;
        valid_corr = isfinite(x_corr) & isfinite(y_corr);
        x_corr = x_corr(valid_corr);
        y_corr = y_corr(valid_corr);
        hold(ax_c,'on');
        scatter(ax_c,x_corr,y_corr,22,'Marker','o','MarkerEdgeColor',this_color,'MarkerFaceColor','w','LineWidth',0.8);
        if numel(x_corr)>=2
            fit_coef = polyfit(x_corr,y_corr,1);
            x_fit = linspace(min(x_corr),max(x_corr),200);
            y_fit = polyval(fit_coef,x_fit);
            plot(ax_c,x_fit,y_fit,'-','Color',this_color,'LineWidth',1.5);
        end
        xlabel(ax_c,'Median distance to pmHG (mm)','FontName','Arial','FontSize',6);
        ylabel(ax_c,'Proportion (%)','FontName','Arial','FontSize',6);
        ylim(ax_c,y_lim);
        if iu==1
            title(ax_c,'Distance-proportion association','FontName','Arial','FontSize',6,'FontWeight','normal');
        end
        star_text = p_to_stars(p_detail(iu));
        stat_text = sprintf('\rho = %.2f, p = %.3g %s',rho_detail(iu),p_detail(iu),star_text);
        if p_detail(iu)<0.05
            stat_color = [0.80 0 0];
        else
            stat_color = [0 0 0];
        end
        text(ax_c,0.04,0.94,stat_text,'Units','normalized','HorizontalAlignment','left','VerticalAlignment','top','FontName','Arial','FontSize',6,'FontWeight','bold','Color',stat_color,'Interpreter','tex');
        set(ax_c,'FontName','Arial','FontSize',6,'Box','off','TickDir','out','LineWidth',0.5,'XColor','k','YColor','k');
        hold(ax_c,'off');
    end

    detail_fig_handles{imode} = h_detail;
end

%% ========================================================================
% 12. SAVE MAIN + DETAIL FIGURES
% =========================================================================

fig_base_raw = fullfile(save_dir,'Hierarchy_12ROI_distanceMedian_unit_proportions_raw');
fig_base_smooth = fullfile(save_dir,'Hierarchy_12ROI_distanceMedian_unit_proportions_smooth');

exportgraphics(main_fig_handles{1},[fig_base_raw '.png'],'Resolution',png_resolution);
exportgraphics(main_fig_handles{1},[fig_base_raw '.pdf'],'ContentType','vector');
exportgraphics(main_fig_handles{2},[fig_base_smooth '.png'],'Resolution',png_resolution);
exportgraphics(main_fig_handles{2},[fig_base_smooth '.pdf'],'ContentType','vector');

detail_base_raw = fullfile(save_dir,'Hierarchy_12ROI_distanceMedian_3x3_detail_raw');
detail_base_smooth = fullfile(save_dir,'Hierarchy_12ROI_distanceMedian_3x3_detail_smooth');

exportgraphics(detail_fig_handles{1},[detail_base_raw '.png'],'Resolution',png_resolution);
exportgraphics(detail_fig_handles{1},[detail_base_raw '.pdf'],'ContentType','vector');
exportgraphics(detail_fig_handles{2},[detail_base_smooth '.png'],'Resolution',png_resolution);
exportgraphics(detail_fig_handles{2},[detail_base_smooth '.pdf'],'ContentType','vector');

%% ========================================================================
% 13. SAVE TABLES
% =========================================================================

writetable( ...
    summary_table, ...
    fullfile(save_dir, ...
    'Hierarchy_12ROI_distanceMedian_summary.csv'));

writetable( ...
    spearman_table, ...
    fullfile(save_dir, ...
    'Hierarchy_12ROI_distanceMedian_Spearman.csv'));

%% ========================================================================
% 14. SAVE SIMPLE ROI ORDER TXT
% =========================================================================

txt_file = fullfile( ...
    save_dir, ...
    'Hierarchy_12ROI_distanceMedian_order.txt');

fid = fopen(txt_file,'w');

if fid<0
    error('Cannot create output TXT:\n%s',txt_file);
end

fprintf(fid, ...
    'Rank\tROI\tMedianDistanceToPmHG_mm\n');

for ir = 1:nROI

    fprintf(fid, ...
        '%d\t%s\t%.6f\n', ...
        ir, ...
        char(summary_table.ROI(ir)), ...
        summary_table.MedianDistanceToPmHG_mm(ir));
end

fclose(fid);

%% ========================================================================
% 15. SAVE MAT
% =========================================================================

save( ...
    fullfile(save_dir, ...
    'Hierarchy_12ROI_distanceMedian_results.mat'), ...
    ...
    'summary_table', ...
    'spearman_table', ...
    'roi_order', ...
    'roi_distance_mm', ...
    ...
    'rho_P','p_P', ...
    'rho_S','p_S', ...
    'rho_Shared','p_Shared', ...
    'enable_smoothing', ...
    'smoothing_method', ...
    'smoothing_window', ...
    'plot_P_raw','plot_S_raw','plot_Shared_raw', ...
    'plot_P_smooth','plot_S_smooth','plot_Shared_smooth');

%% ========================================================================
% 16. FINAL CONSOLE SUMMARY
% =========================================================================

fprintf('\n============================================================\n');
fprintf('FINAL HIERARCHY SUMMARY\n');
fprintf('============================================================\n');

fprintf('P-only : rho = %.4f, p = %.6g\n',rho_P,p_P);
fprintf('S-only : rho = %.4f, p = %.6g\n',rho_S,p_S);
fprintf('Shared : rho = %.4f, p = %.6g\n',rho_Shared,p_Shared);

fprintf('\nFigure smoothing: %s\n',string(enable_smoothing));
if enable_smoothing
    fprintf('Smoothing method: %s\n',smoothing_method);
    fprintf('Smoothing window: %d ROIs\n',smoothing_window);
end
fprintf('NOTE: Spearman statistics use raw unsmoothed proportions.\n');
fprintf('Both raw and smoothed figures were exported.\n');

fprintf('\nSaved to:\n%s\n',save_dir);

%% ========================================================================
% LOCAL FUNCTION 1: P VALUE TO SIGNIFICANCE STARS
% =========================================================================

function star_text = p_to_stars(p)

if ~isfinite(p)

    star_text = 'n.s.';

elseif p<0.001

    star_text = '***';

elseif p<0.01

    star_text = '**';

elseif p<0.05

    star_text = '*';

else

    star_text = 'n.s.';
end

end

%% ========================================================================
% LOCAL FUNCTION 2: MAP DETAILED ANATOMY TO FINAL 12 ROIs
% =========================================================================

function roi = map_brain_region_to_roi12(region)

s = lower(strtrim(string(region)));
roi = "";

if strlength(s)==0
    return;
end

% HG
if contains(s,'transverse temporal gyrus') || ...
        contains(s,'heschl')

    roi = "HG";

% pmSTG
elseif contains(s,'superior temporal gyrus - middle') || ...
        contains(s,'superior temporal gyrus - posterior') || ...
        contains(s,'middle superior temporal gyrus') || ...
        contains(s,'posterior superior temporal gyrus')

    roi = "pmSTG";

% Insula
elseif contains(s,'insula')

    roi = "Insula";

% Sensorimotor
elseif contains(s,'paracentral') || ...
        contains(s,'postcentral') || ...
        contains(s,'precentral')

    roi = "Sensorimotor";

% aSTG
elseif contains(s,'superior temporal gyrus - anterior') || ...
        contains(s,'anterior superior temporal gyrus')

    roi = "aSTG";

% MTG
elseif contains(s,'middle temporal gyrus')

    roi = "MTG";

% Temporal Pole
elseif contains(s,'temporal pole')

    roi = "Temporal Pole";

% ITG
elseif contains(s,'fusiform gyrus') || ...
        contains(s,'inferior temporal gyrus')

    roi = "ITG";

% IPL
elseif contains(s,'angular gyrus') || ...
        contains(s,'supramarginal gyrus')

    roi = "IPL";

% PCC
elseif contains(s,'cingulate gyrus - posterior') || ...
        contains(s,'posterior cingulate')

    roi = "PCC";

% Precuneus
elseif contains(s,'precuneus')

    roi = "Precuneus";

% MTL
elseif contains(s,'amygdala') || ...
        contains(s,'hippocampus') || ...
        contains(s,'parahippocampal')

    roi = "MTL";

% Excluded:
%   Cuneus -> visual/occipital
%   MFG, IFG -> insufficient responsive-participant coverage
%   all other labels remain unmapped
end

end
