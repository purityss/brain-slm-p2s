%% step06_plot_lesion_results_from_summary_excel.m
% =========================================================================
% Plot targeted lesion results directly from per-model Excel summaries.
%
% INPUT
% -----
% code_for_SLM_data/
%   code_for_all_models_group/step06_lesion_results/lesion_summary/
%       <model>.xlsx
%
% Each Excel file is expected to contain:
%
%   Sheet "lesion_accuracy"
%       lesion_count | p_only | s_only | shared
%
%   Sheet "delta_pp"
%       lesion_count | p_only | s_only | shared
%
%   Sheet "metadata"
%       field | value
%
% Relevant metadata:
%       model_key
%       maximum_equal_N
%
% MAXIMUM LESION N
% ----------------
% For each model:
%
%   onePercentN = floor(0.01 * total model units)
%   maximumEqualN = metadata.maximum_equal_N
%                 = maximum matched N permitted by the three unit classes
%
%   maxLesionN = min(onePercentN, maximumEqualN)
%
% Only rows with lesion_count <= maxLesionN are plotted.
% The x-axis ALWAYS extends to maxLesionN and labels that exact value.
%
% Example:
%   Baichuan-Audio-Base:
%       total units       = 147,712
%       1%                = 1,477
%       maximum_equal_N   = 1,697
%       final maxLesionN  = 1,477
%
% PERFORMANCE
% -----------
% "lesion_accuracy" stores model accuracy as a fraction (0-1).
% Curves display accuracy * 100 (%).
%
% "delta_pp" already stores change from intact baseline in percentage points.
% Bars therefore use delta_pp directly.
%
% BAR N
% -----
% Prefer the exact final maxLesionN row.
% If that exact row is absent, use the largest available lesion_count <=
% maxLesionN for which all three conditions have finite values.
%
% OUTPUT
% ------
% step06_lesion_results\matlab_results\
%     lesion_maxN_summary.xlsx
%     lesion_curves_all_models.pdf/.png
%     lesion_bars_all_models.pdf/.png
%     curves_individual\<model>_lesion_curve.pdf/.png
%
% FIGURE STYLE
% ------------
% Arial 7 pt, all text black, axes 0.5 pt black, white background.
% Each model has its OWN y-axis range.
% =========================================================================

clearvars;
close all;
clc;

%% ============================ USER SETTINGS ==============================
thisFile = mfilename('fullpath');
if isempty(thisFile)
    error('Please save this script before running it.');
end

step06Root = fileparts(thisFile);

lesionSummaryRoot = fullfile(step06Root,'lesion_summary');
outputRoot = fullfile(step06Root,'matlab_results');
individualCurveRoot = fullfile(outputRoot,'curves_individual');

if ~isfolder(lesionSummaryRoot)
    error('Cannot find lesion_summary folder:\n%s',lesionSummaryRoot);
end
if ~isfolder(outputRoot), mkdir(outputRoot); end
if ~isfolder(individualCurveRoot), mkdir(individualCurveRoot); end

summaryExcel = fullfile(outputRoot,'lesion_maxN_summary.xlsx');

savePDF = true;
savePNG = true;
pngResolution = 600;

% Colors.
PColor      = [180 37 35]/255;
SColor      = [34 91 142]/255;
SharedColor = [103 38 141]/255;

% Global style.
set(groot, ...
    'defaultAxesFontName','Arial', ...
    'defaultTextFontName','Arial', ...
    'defaultAxesFontSize',7, ...
    'defaultTextFontSize',7, ...
    'defaultAxesLineWidth',0.5, ...
    'defaultAxesXColor','k', ...
    'defaultAxesYColor','k', ...
    'defaultTextColor','k');

%% ============================== MODEL TABLE ==============================
Model = [ ...
    "Baichuan-Audio-Base"
    "Baichuan-Audio-Instruct"
    "Covo-Audio-Chat"
    "FireRedAudio"
    "Freeze-Omni"
    "GLM-4-Voice"
    "Kimi-Audio-7B-Instruct"
    "MiniCPM-o 2.6"
    "MiniCPM-o 4.5"
    "Qwen-Audio"
    "Qwen-Audio-Chat"
    "Qwen2.5-Omni"
    "Qwen2-Audio"
    "Qwen2-Audio-Instruct"];

ModelSlug = [ ...
    "baichuan_audio_base"
    "baichuan_audio_instruct"
    "covo_audio_chat"
    "firered_audio"
    "freeze_omni"
    "glm_4_voice"
    "kimi_audio_7b_instruct"
    "minicpm_o_2_6"
    "minicpm_o_4_5"
    "qwen_audio"
    "qwen_audio_chat"
    "qwen2_5_omni"
    "qwen2_audio"
    "qwen2_audio_instruct"];

TotalUnits = [ ...
    147712
    147712
    147456
    178688
    130560
    188160
    147456
    130560
    178176
    178688
    178688
    147456
    178688
    178688];

OnePercentRaw = 0.01 * TotalUnits;
OnePercentN = floor(OnePercentRaw);

Registry = table(Model,ModelSlug,TotalUnits,OnePercentRaw,OnePercentN);

%% ======================== INDEX SUMMARY EXCEL FILES =====================
SummaryFiles = index_lesion_summary_excels(lesionSummaryRoot);

fprintf('\n============================================================\n');
fprintf('LESION SUMMARY EXCEL FILES\n');
fprintf('============================================================\n');

for i = 1:height(SummaryFiles)
    fprintf('%-28s %s\n',SummaryFiles.ModelKey(i),SummaryFiles.File(i));
end

%% ============================ READ ALL MODELS ============================
conditions = ["p_only" "s_only" "shared"];
conditionLabels = ["Phonological units" "Semantic units" "Shared units"];

MaximumEqualN = nan(height(Registry),1);
ClassMinN = nan(height(Registry),1);
TheoreticalMaxN = nan(height(Registry),1);
LimitRule = strings(height(Registry),1);
BaselineAccuracy = nan(height(Registry),1);
BaselinePerformancePct = nan(height(Registry),1);
CurveMaxAvailableN = nan(height(Registry),1);
BarUsedN = nan(height(Registry),1);
ExactMaxNAvailable = false(height(Registry),1);

PBarChange = nan(height(Registry),1);
SBarChange = nan(height(Registry),1);
SharedBarChange = nan(height(Registry),1);

ResolvedSummaryFile = strings(height(Registry),1);

CurveData = table();
BarData = table();

for im = 1:height(Registry)
    modelName = Registry.Model(im);
    slug = Registry.ModelSlug(im);

    fprintf('\n[%d/%d] %s\n',im,height(Registry),modelName);

    summaryFile = resolve_summary_excel(SummaryFiles,slug);
    ResolvedSummaryFile(im) = string(summaryFile);

    % ----------------------------- metadata ------------------------------
    meta = read_metadata_sheet(summaryFile);

    if ~isfield(meta,'model_key')
        error('%s: metadata sheet does not contain model_key.',summaryFile);
    end

    excelKey = normalize_model_key(string(meta.model_key));
    targetKey = normalize_model_key(slug);

    if excelKey ~= targetKey
        error(['Model key mismatch:\n' ...
            'Registry: %s\nExcel metadata: %s\nFile: %s'], ...
            slug,string(meta.model_key),summaryFile);
    end

    if ~isfield(meta,'maximum_equal_N')
        error('%s: metadata sheet does not contain maximum_equal_N.',summaryFile);
    end

    MaximumEqualN(im) = str2double(string(meta.maximum_equal_N));

    if ~isfinite(MaximumEqualN(im))
        error('%s: invalid maximum_equal_N.',summaryFile);
    end

    MaximumEqualN(im) = floor(MaximumEqualN(im));
    ClassMinN(im) = MaximumEqualN(im);
    if isfield(meta,'class_min_N')
        ClassMinN(im) = floor(str2double(string(meta.class_min_N)));
    end
    TheoreticalMaxN(im) = min(Registry.OnePercentN(im),ClassMinN(im));
    if MaximumEqualN(im) ~= TheoreticalMaxN(im)
        error('%s: maximum_equal_N does not match the final 1%%/class cap.',summaryFile);
    end

    if Registry.OnePercentN(im) < ClassMinN(im)
        LimitRule(im) = "1% of all units";
    elseif ClassMinN(im) < Registry.OnePercentN(im)
        LimitRule(im) = "minimum of three unit classes";
    else
        LimitRule(im) = "tie";
    end

    fprintf('  total units      = %d\n',Registry.TotalUnits(im));
    fprintf('  1%% upper N       = %d\n',Registry.OnePercentN(im));
    fprintf('  maximum_equal_N  = %d\n',MaximumEqualN(im));
    fprintf('  final max N      = %d (%s)\n',TheoreticalMaxN(im),LimitRule(im));

    % -------------------------- lesion accuracy --------------------------
    A = readtable(summaryFile, ...
        'Sheet','lesion_accuracy', ...
        'VariableNamingRule','preserve');

    require_columns(A,{'lesion_count','p_only','s_only','shared'}, ...
        summaryFile,'lesion_accuracy');

    A = A(:,{'lesion_count','p_only','s_only','shared'});
    A.lesion_count = double(A.lesion_count);
    A.p_only = double(A.p_only);
    A.s_only = double(A.s_only);
    A.shared = double(A.shared);

    validRows = isfinite(A.lesion_count);
    A = A(validRows,:);
    A = sortrows(A,'lesion_count');

    % Never use results above the final model-specific limit.
    Aplot = A(A.lesion_count <= TheoreticalMaxN(im),:);

    if isempty(Aplot)
        error('%s: lesion_accuracy has no rows <= max N.',modelName);
    end

    % Baseline = lesion_count 0. It should be identical across three columns.
    i0 = find(Aplot.lesion_count==0,1,'first');
    if isempty(i0)
        error('%s: lesion_accuracy does not contain lesion_count = 0.',modelName);
    end

    baseValues = [Aplot.p_only(i0) Aplot.s_only(i0) Aplot.shared(i0)];
    if any(~isfinite(baseValues))
        error('%s: invalid baseline accuracy.',modelName);
    end

    BaselineAccuracy(im) = mean(baseValues);
    BaselinePerformancePct(im) = BaselineAccuracy(im)*100;

    nonzeroDoses = Aplot.lesion_count(Aplot.lesion_count>0);
    if ~isempty(nonzeroDoses)
        CurveMaxAvailableN(im) = max(nonzeroDoses);
    end

    exactRow = find(Aplot.lesion_count==TheoreticalMaxN(im),1,'first');
    ExactMaxNAvailable(im) = ~isempty(exactRow);

    % Save long-form curve data.
    for ic = 1:3
        cond = conditions(ic);
        condLabel = conditionLabels(ic);
        y = Aplot.(char(cond));

        for ir = 1:height(Aplot)
            n = Aplot.lesion_count(ir);
            acc = y(ir);

            if ~isfinite(acc)
                continue;
            end

            row = table( ...
                modelName,slug,cond,condLabel,n,acc,acc*100, ...
                'VariableNames',{ ...
                'Model','ModelSlug','Condition','ConditionLabel', ...
                'LesionCount','Accuracy','PerformancePercent'});

            CurveData = [CurveData;row]; %#ok<AGROW>
        end
    end

    % ----------------------------- delta pp ------------------------------
    D = readtable(summaryFile, ...
        'Sheet','delta_pp', ...
        'VariableNamingRule','preserve');

    require_columns(D,{'lesion_count','p_only','s_only','shared'}, ...
        summaryFile,'delta_pp');

    D = D(:,{'lesion_count','p_only','s_only','shared'});
    D.lesion_count = double(D.lesion_count);
    D.p_only = double(D.p_only);
    D.s_only = double(D.s_only);
    D.shared = double(D.shared);

    validRows = isfinite(D.lesion_count);
    D = D(validRows,:);
    D = sortrows(D,'lesion_count');
    D = D(D.lesion_count <= TheoreticalMaxN(im),:);

    finiteAll = isfinite(D.p_only) & isfinite(D.s_only) & isfinite(D.shared);
    candidateRows = find(D.lesion_count>0 & finiteAll);

    if isempty(candidateRows)
        warning('%s: no common nonzero delta_pp row <= max N.',modelName);
        continue;
    end

    % Prefer exact final max N; otherwise largest available common N below it.
    exactDeltaRow = candidateRows(D.lesion_count(candidateRows)==TheoreticalMaxN(im));

    if ~isempty(exactDeltaRow)
        ibar = exactDeltaRow(1);
    else
        [~,k] = max(D.lesion_count(candidateRows));
        ibar = candidateRows(k);
    end

    BarUsedN(im) = D.lesion_count(ibar);

    PBarChange(im) = D.p_only(ibar);
    SBarChange(im) = D.s_only(ibar);
    SharedBarChange(im) = D.shared(ibar);

    barRow = table( ...
        modelName,slug,BarUsedN(im), ...
        PBarChange(im),SBarChange(im),SharedBarChange(im), ...
        'VariableNames',{ ...
        'Model','ModelSlug','BarUsedN', ...
        'PonlyChangePctPoint','SonlyChangePctPoint','SharedChangePctPoint'});

    BarData = [BarData;barRow]; %#ok<AGROW>
end

%% ============================ SUMMARY TABLE ==============================
Summary = Registry;
Summary.MaximumEqualN = MaximumEqualN;
Summary.TheoreticalMaxN = TheoreticalMaxN;
Summary.LimitRule = LimitRule;
Summary.CurveMaxAvailableN = CurveMaxAvailableN;
Summary.ExactMaxNAvailable = ExactMaxNAvailable;
Summary.BarUsedN = BarUsedN;
Summary.BaselineAccuracy = BaselineAccuracy;
Summary.BaselinePerformancePercent = BaselinePerformancePct;
Summary.PonlyBarChangePctPoint = PBarChange;
Summary.SonlyBarChangePctPoint = SBarChange;
Summary.SharedBarChangePctPoint = SharedBarChange;
Summary.SummaryExcelFile = ResolvedSummaryFile;

if isfile(summaryExcel)
    delete(summaryExcel);
end

writetable(Summary,summaryExcel,'Sheet','MaxN_Summary');
writetable(CurveData,summaryExcel,'Sheet','CurveData');
writetable(BarData,summaryExcel,'Sheet','BarData');

fprintf('\nSaved summary Excel:\n%s\n',summaryExcel);

%% ========================= CURVE FIGURE: ALL MODELS ======================
figCurve = figure( ...
    'Color','w', ...
    'Position',[100 60 680 560], ...
    'Renderer','painters');

tl = tiledlayout(figCurve,4,4, ...
    'TileSpacing','compact', ...
    'Padding','compact');

for im = 1:height(Registry)
    ax = nexttile(tl,im);
    hold(ax,'on');

    slug = Registry.ModelSlug(im);
    colors = [PColor;SColor;SharedColor];

    for ic = 1:3
        use = CurveData.ModelSlug==slug & CurveData.Condition==conditions(ic);
        T = sortrows(CurveData(use,:),'LesionCount');

        if isempty(T)
            continue;
        end

        % plot(ax,T.LesionCount,T.PerformancePercent,'-o', ...
        %     'Color',colors(ic,:), ...
        %     'MarkerEdgeColor',colors(ic,:), ...
        %     'MarkerFaceColor',colors(ic,:), ...
        %     'LineWidth',1.0, ...
        %     'MarkerSize',2.5);

        markerList = {'o','s','d'};  % P-only圆形，S-only方形，Shared菱形
plot(ax,T.LesionCount,T.PerformancePercent, ...
    ['-' markerList{ic}], ...
    'Color',colors(ic,:), ...
    'MarkerEdgeColor',colors(ic,:), ...
    'MarkerFaceColor',colors(ic,:), ...
    'LineWidth',1.0, ...
    'MarkerSize',2.5);

    end

    % Exact model-specific final limit and explicitly displayed tick values.
    xMax = TheoreticalMaxN(im);
    xlim(ax,[0 xMax]);
    xTickValues = lesion_xticks(xMax);
    xticks(ax,xTickValues);
    xticklabels(ax,string(xTickValues));

    % Model-specific y-axis using ONLY data plotted for this model.
    modelRows = CurveData.ModelSlug==slug;
    yAll = CurveData.PerformancePercent(modelRows);
    ylim(ax,performance_ylim(yAll));

    title(ax,Registry.Model(im), ...
        'Interpreter','none', ...
        'FontName','Arial','FontSize',7, ...
        'Color','k');

    style_axes(ax);
    hold(ax,'off');
end

% Shared axis titles for the all-model lesion-curve figure.
xlabel(tl,'Number of lesioned model units', ...
    'FontName','Arial','FontSize',7,'Color','k');
ylabel(tl,'Model performance in Fleurs-zh test set (%)', ...
    'FontName','Arial','FontSize',7,'Color','k');

% Shared legend.
axLegend = nexttile(tl,15,[1 2]);
hold(axLegend,'on');

h1 = plot(axLegend,nan,nan,'-o', ...
    'Color',PColor,'MarkerFaceColor',PColor,'MarkerEdgeColor',PColor, ...
    'LineWidth',1,'MarkerSize',2.5);

h2 = plot(axLegend,nan,nan,'-s', ...
    'Color',SColor,'MarkerFaceColor',SColor,'MarkerEdgeColor',SColor, ...
    'LineWidth',1,'MarkerSize',2.5);

h3 = plot(axLegend,nan,nan,'-d', ...
    'Color',SharedColor,'MarkerFaceColor',SharedColor,'MarkerEdgeColor',SharedColor, ...
    'LineWidth',1,'MarkerSize',2.5);

legend(axLegend,[h1 h2 h3], ...
    {'Phonological units','Semantic units','Shared units'}, ...
    'Orientation','horizontal', ...
    'Location','north', ...
    'Box','off', ...
    'FontName','Arial', ...
    'FontSize',7, ...
    'TextColor','k');

axis(axLegend,'off');
hold(axLegend,'off');

export_both(figCurve,outputRoot,'lesion_curves_all_models', ...
    savePDF,savePNG,pngResolution);
close(figCurve);

%% ======================== INDIVIDUAL CURVE FIGURES =======================
for im = 1:height(Registry)
    fig = figure( ...
        'Color','w', ...
        'Position',[100 100 230 175], ...
        'Renderer','painters');

    ax = axes(fig,'Position',[0.20 0.20 0.74 0.70]);
    hold(ax,'on');

    slug = Registry.ModelSlug(im);
    colors = [PColor;SColor;SharedColor];
    h = gobjects(3,1);

    for ic = 1:3
        use = CurveData.ModelSlug==slug & CurveData.Condition==conditions(ic);
        T = sortrows(CurveData(use,:),'LesionCount');

        if isempty(T)
            continue;
        end

        h(ic) = plot(ax,T.LesionCount,T.PerformancePercent,['-' markerList{ic}], ...
            'Color',colors(ic,:), ...
            'MarkerEdgeColor',colors(ic,:), ...
            'MarkerFaceColor',colors(ic,:), ...
            'LineWidth',1.0, ...
            'MarkerSize',2.8);
    end

    xMax = TheoreticalMaxN(im);
    xlim(ax,[0 xMax]);
    xticks(ax,lesion_xticks(xMax));

    modelRows = CurveData.ModelSlug==slug;
    yAll = CurveData.PerformancePercent(modelRows);
    ylim(ax,performance_ylim(yAll));

    xlabel(ax,'Lesioned units', ...
        'FontName','Arial','FontSize',7,'Color','k');

    ylabel(ax,'Model performance (%)', ...
        'FontName','Arial','FontSize',7,'Color','k');

    title(ax,Registry.Model(im), ...
        'Interpreter','none', ...
        'FontName','Arial','FontSize',7, ...
        'Color','k');

    validH = isgraphics(h);
    legendLabels = {'Phonological','Semantic','Shared'};
    legend(ax,h(validH), ...
        legendLabels(validH), ...
        'Location','best', ...
        'Box','off', ...
        'FontName','Arial', ...
        'FontSize',7, ...
        'TextColor','k');

    style_axes(ax);
    hold(ax,'off');

    export_both(fig,individualCurveRoot, ...
        char(Registry.ModelSlug(im)+"_lesion_curve"), ...
        savePDF,savePNG,pngResolution);

    close(fig);
end

%% ============================== BAR FIGURE ================================
figBar = figure( ...
    'Color','w', ...
    'Position',[100 60 680 560], ...
    'Renderer','painters');

tlb = tiledlayout(figBar,4,4, ...
    'TileSpacing','compact', ...
    'Padding','compact');

barColors = [PColor;SColor;SharedColor];

for im = 1:height(Registry)
    ax = nexttile(tlb,im);
    hold(ax,'on');

    vals = [PBarChange(im) SBarChange(im) SharedBarChange(im)];

    if all(~isfinite(vals))
        text(ax,0.5,0.5,'No matched dose', ...
            'Units','normalized', ...
            'HorizontalAlignment','center', ...
            'FontName','Arial','FontSize',7,'Color','k');
        axis(ax,'off');
        continue;
    end

    b = bar(ax,1:3,vals,0.58, ...
        'FaceColor','flat', ...
        'EdgeColor','none');

    b.CData = barColors;

    yline(ax,0,'-','Color','k','LineWidth',0.5);

    xlim(ax,[0.45 3.55]);
    xticks(ax,[]);
    ylim(ax,change_ylim(vals));

    if mod(im-1,4)==0
        ylabel(ax,{'Change in model','performance (%)'}, ...
            'FontName','Arial','FontSize',7,'Color','k');
    end

    title(ax,Registry.Model(im), ...
        'Interpreter','none', ...
        'FontName','Arial','FontSize',7, ...
        'Color','k');

    style_axes(ax);
    hold(ax,'off');
end

% Shared legend.
axLegend = nexttile(tlb,15,[1 2]);
hold(axLegend,'on');

h1 = bar(axLegend,nan,'FaceColor',PColor,'EdgeColor','none');
h2 = bar(axLegend,nan,'FaceColor',SColor,'EdgeColor','none');
h3 = bar(axLegend,nan,'FaceColor',SharedColor,'EdgeColor','none');

legend(axLegend,[h1 h2 h3], ...
    {'Phonological units','Semantic units','Shared units'}, ...
    'Orientation','horizontal', ...
    'Location','north', ...
    'Box','off', ...
    'FontName','Arial', ...
    'FontSize',7, ...
    'TextColor','k');

axis(axLegend,'off');
hold(axLegend,'off');

export_both(figBar,outputRoot,'lesion_bars_all_models', ...
    savePDF,savePNG,pngResolution);
close(figBar);

fprintf('\n============================================================\n');
fprintf('LESION PLOTTING COMPLETE\n');
fprintf('============================================================\n');
fprintf('Input:  %s\n',lesionSummaryRoot);
fprintf('Output: %s\n',outputRoot);

%% ============================== FUNCTIONS ================================

function F = index_lesion_summary_excels(root)
% Read model_key from every Excel metadata sheet and build a lookup table.

files = dir(fullfile(root,'*.xlsx'));

if isempty(files)
    error('No .xlsx files found in:\n%s',root);
end

keys = strings(numel(files),1);
paths = strings(numel(files),1);

for i = 1:numel(files)
    path = fullfile(files(i).folder,files(i).name);
    meta = read_metadata_sheet(path);

    if ~isfield(meta,'model_key')
        error('metadata.model_key missing in:\n%s',path);
    end

    keys(i) = normalize_model_key(string(meta.model_key));
    paths(i) = string(path);
end

if numel(unique(keys)) ~= numel(keys)
    error('Duplicate model_key values found in lesion_summary Excel files.');
end

F = table(keys,paths,'VariableNames',{'ModelKey','File'});
end

function file = resolve_summary_excel(F,slug)
key = normalize_model_key(slug);
idx = find(F.ModelKey==key);

if isempty(idx)
    error('No lesion_summary Excel found for model: %s',slug);
elseif numel(idx)>1
    error('Multiple lesion_summary Excel files found for model: %s',slug);
end

file = char(F.File(idx));
end

function meta = read_metadata_sheet(file)
C = readcell(file,'Sheet','metadata');

if size(C,2)<2
    error('metadata sheet must contain at least two columns:\n%s',file);
end

% Locate field/value header.
headerRow = [];
for r = 1:size(C,1)
    if string(C{r,1})=="field" && string(C{r,2})=="value"
        headerRow = r;
        break;
    end
end

if isempty(headerRow)
    error('Could not find field/value header in metadata sheet:\n%s',file);
end

meta = struct();

for r = headerRow+1:size(C,1)
    if isempty(C{r,1})
        continue;
    end

    field = matlab.lang.makeValidName(char(string(C{r,1})));
    meta.(field) = C{r,2};
end
end

function require_columns(T,names,file,sheetName)
missing = setdiff(string(names),string(T.Properties.VariableNames));

if ~isempty(missing)
    error(['Missing columns in %s [%s]: %s'], ...
        file,sheetName,strjoin(missing,', '));
end
end

function key = normalize_model_key(x)
key = lower(string(x));
key = regexprep(key,'[^a-z0-9]+','_');
key = regexprep(key,'^_+|_+$','');
end

function ticks = lesion_xticks(maxN)
% Compact major ticks while ALWAYS labeling the exact model-specific maximum.
maxN = double(maxN);

if ~isfinite(maxN) || maxN<=0
    ticks = 0;
    return;
end

if maxN >= 500
    regularTicks = 0:500:(floor(maxN/500)*500);
    ticks = unique([regularTicks maxN]);
else
    mid = round((maxN/2)/50)*50;

    if mid<=0 || mid>=maxN
        ticks = [0 maxN];
    else
        ticks = [0 mid maxN];
    end
end
end

function yl = performance_ylim(y)
% Model-specific y-axis with modest padding and clean rounded bounds.

y = double(y(:));
y = y(isfinite(y));

if isempty(y)
    yl = [0 100];
    return;
end

lo = min(y);
hi = max(y);

if hi-lo < 1
    pad = 1;
else
    pad = 0.10*(hi-lo);
end

lo = max(0,lo-pad);
hi = min(100,hi+pad);

% Round outward to clean 5-percentage-point limits.
lo = floor(lo/5)*5;
hi = ceil(hi/5)*5;

if hi<=lo
    lo = max(0,lo-5);
    hi = min(100,hi+5);
end

yl = [lo hi];
end

function yl = change_ylim(vals)
% Independent y-axis for each model's bar panel.

vals = double(vals(:));
vals = vals(isfinite(vals));

if isempty(vals)
    yl = [-1 1];
    return;
end

lo = min([vals;0]);
hi = max([vals;0]);

span = hi-lo;

if span<=eps
    pad = 1;
else
    pad = max(0.08*span,0.15);
end

lo = lo-pad;
hi = hi+pad;

% Round outward; use finer rounding for small effects.
mx = max(abs([lo hi]));

if mx <= 1
    step = 0.2;
elseif mx <= 5
    step = 1;
elseif mx <= 20
    step = 5;
else
    step = 10;
end

lo = floor(lo/step)*step;
hi = ceil(hi/step)*step;

if hi<=lo
    hi = lo+step;
end

yl = [lo hi];
end

function style_axes(ax)
set(ax, ...
    'FontName','Arial', ...
    'FontSize',7, ...
    'FontWeight','normal', ...
    'LineWidth',0.5, ...
    'XColor','k', ...
    'YColor','k', ...
    'TickDir','out', ...
    'Box','off', ...
    'Color','w');

% grid(ax,'off');
grid(ax,'on');
set(ax, ...
    'GridColor',[0.8 0.8 0.8], ...
    'GridAlpha',0.5, ...
    'GridLineStyle','-');
end

function export_both(fig,outDir,name,savePDF,savePNG,resolution)
if savePDF
    exportgraphics(fig, ...
        fullfile(outDir,[name '.pdf']), ...
        'ContentType','vector');
end

if savePNG
    exportgraphics(fig, ...
        fullfile(outDir,[name '.png']), ...
        'Resolution',resolution);
end
end
