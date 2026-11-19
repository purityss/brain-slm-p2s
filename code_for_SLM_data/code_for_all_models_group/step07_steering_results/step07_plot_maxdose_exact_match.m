%% step07_plot_maxdose_exact_match.m
% =========================================================================
% Plot finalized Step07 maximum-dose steering results for BOTH datasets.
%
% Metric
% ------
% The y-axis is exact-match accuracy:
%       100 * number of completely correct utterances / number of samples.
% Every marker is labelled as "correct/total". Shared is additionally
% labelled by its integer exact-count change relative to baseline.
%
% Default models
% --------------
%       baichuan_audio_base, qwen2_audio
% Edit the two selected-model lists below to add, remove, or reorder models.
%
% Inputs
% ------
% 52words/summary/all_models_summary.csv
% fleurs_zh/summary/all_models_summary.csv
%
% Outputs
% -------
% matlab_results/step07_52words_maxdose_exact_match.pdf/.png
% matlab_results/step07_fleurs_zh_maxdose_exact_match.pdf/.png
%
% Visual encoding exactly follows Step06 lesion plotting:
% P-only = red circle; S-only = blue square; Shared = purple diamond.
% =========================================================================

clearvars;
close all;
clc;

%% ============================ USER SETTINGS =============================
thisFile = mfilename('fullpath');
if isempty(thisFile)
    error('Please save this script before running it.');
end

resultsRoot = fileparts(thisFile);
outputRoot = fullfile(resultsRoot,'matlab_results');
if ~isfolder(outputRoot), mkdir(outputRoot); end

% Add, remove, or reorder model keys independently for each dataset.
selectedModels52words = [ ...
    "baichuan_audio_base"
    "qwen2_audio"];

selectedModelsFleurs = [ ...
    "baichuan_audio_base"
    "qwen2_audio"];

savePDF = true;
savePNG = true;
pngResolution = 600;

% Leave empty for a truthful data-adaptive range. To force a shared range,
% specify, for example, forcedYLim = [20 90].
forcedYLim = [];

% Step06 lesion colors and markers (do not change across figures).
PColor      = [180 37 35]/255;
SColor      = [34 91 142]/255;
SharedColor = [103 38 141]/255;

set(groot, ...
    'defaultAxesFontName','Arial', ...
    'defaultTextFontName','Arial', ...
    'defaultAxesFontSize',7, ...
    'defaultTextFontSize',7, ...
    'defaultAxesLineWidth',0.5, ...
    'defaultAxesXColor','k', ...
    'defaultAxesYColor','k', ...
    'defaultTextColor','k');

%% ============================ DRAW BOTH DATASETS =========================
datasets = ["52words" "fleurs_zh"];
datasetTitles = ["52 words" "FLEURS-zh"];
datasetModels = {selectedModels52words,selectedModelsFleurs};

for id = 1:numel(datasets)
    dataset = datasets(id);
    selectedModels = datasetModels{id};
    summaryFile = resolve_summary_file(resultsRoot,dataset,selectedModels);
    endpoints = read_endpoint_table(summaryFile,selectedModels);

    fprintf('\n[%s] Input: %s\n',dataset,summaryFile);
    print_endpoint_values(endpoints,selectedModels);

    fig = draw_exact_figure( ...
        endpoints,selectedModels,datasetTitles(id),forcedYLim, ...
        PColor,SColor,SharedColor);

    stem = "step07_" + dataset + "_maxdose_exact_match";
    export_figure(fig,outputRoot,char(stem),savePDF,savePNG,pngResolution);
    close(fig);
end

fprintf('\nStep07 exact-match figures saved to:\n%s\n',outputRoot);

%% ================================ FUNCTIONS =============================

function file = resolve_summary_file(root,dataset,selectedModels)
% Require a complete endpoint table for every selected model and condition.

if dataset=="52words"
    candidates = string(fullfile(root,'52words','summary', ...
        'all_models_summary.csv'));
else
    candidates = string(fullfile(root,'fleurs_zh','summary','all_models_summary.csv'));
end

diagnostics = strings(0,1);
for i = 1:numel(candidates)
    candidate = candidates(i);
    if ~isfile(candidate)
        diagnostics(end+1) = candidate + " (missing)"; %#ok<AGROW>
        continue;
    end
    try
        E = read_endpoint_table(candidate,selectedModels);
        if has_complete_conditions(E,selectedModels)
            file = char(candidate);
            return;
        end
        diagnostics(end+1) = candidate + " (incomplete selected conditions)"; %#ok<AGROW>
    catch ME
        diagnostics(end+1) = candidate + " (" + string(ME.message) + ")"; %#ok<AGROW>
    end
end

error('No complete Step07 endpoint table was found for %s:\n%s', ...
    dataset,strjoin(diagnostics,newline));
end

function E = read_endpoint_table(file,selectedModels)
% Normalize the two maintained CSV schemas to one endpoint table.

T = readtable(file,'VariableNamingRule','preserve','TextType','string');
names = string(T.Properties.VariableNames);

modelVar = choose_variable(names,["model_key" "model"]);
classVar = choose_variable(names,["condition" "class" "unit_class"]);
nVar = choose_variable(names,["N" "dose" "n_units" "max_n"]);
samplesVar = choose_variable(names,["samples" "n_samples" "total"]);
baselineVar = choose_variable(names,["baseline_exact" "baseline_exact_count"]);
steeredVar = choose_variable(names,["condition_exact" "steered_exact" "exact"]);

Model = normalize_key(string(T.(modelVar)));
Condition = normalize_condition(string(T.(classVar)));
N = numeric_column(T.(nVar),nVar);
Samples = numeric_column(T.(samplesVar),samplesVar);
BaselineExact = numeric_column(T.(baselineVar),baselineVar);
SteeredExact = numeric_column(T.(steeredVar),steeredVar);

E = table(Model,Condition,N,Samples,BaselineExact,SteeredExact);
E = E(ismember(E.Model,normalize_key(selectedModels)),:);

if isempty(E)
    error('No selected models were found in %s.',file);
end

% Endpoint tables must contain exactly one row per model/condition.
[groups,modelGroup,classGroup] = findgroups(E.Model,E.Condition);
counts = splitapply(@numel,E.Model,groups);
duplicate = counts>1;
if any(duplicate)
    bad = modelGroup(duplicate) + "/" + classGroup(duplicate);
    error('Duplicate endpoint rows in %s: %s',file,strjoin(bad,', '));
end

if any(~isfinite(E.N) | ~isfinite(E.Samples) | E.Samples<=0 | ...
        ~isfinite(E.BaselineExact) | ~isfinite(E.SteeredExact))
    error('Non-finite endpoint counts found in %s.',file);
end
end

function tf = has_complete_conditions(E,selectedModels)
conditions = ["p_only" "s_only" "shared"];
tf = true;
for i = 1:numel(selectedModels)
    model = normalize_key(selectedModels(i));
    observed = E.Condition(E.Model==model);
    if ~all(ismember(conditions,observed))
        tf = false;
        return;
    end
end
end

function fig = draw_exact_figure(E,selectedModels,datasetTitle,forcedYLim, ...
        PColor,SColor,SharedColor)

conditions = ["p_only" "s_only" "shared"];
conditionLabels = {'P-only','S-only','Shared'};
colors = [PColor;SColor;SharedColor];
markers = {'o','s','d'};
offsets = [-0.33 -0.11 0.11 0.33];

nModels = numel(selectedModels);
fig = figure( ...
    'Color','w', ...
    'Position',[100 100 230 220], ...
    'Renderer','painters');
ax = axes(fig,'Position',[0.22 0.33 0.74 0.55]);
hold(ax,'on');

hBaseline = scatter(ax,nan,nan,24,'x', ...
    'MarkerEdgeColor',[0.12 0.12 0.12],'LineWidth',1.0);
hConditions = gobjects(3,1);
allPercent = [];

for im = 1:nModels
    model = normalize_key(selectedModels(im));
    rows = E(E.Model==model,:);
    if isempty(rows)
        error('Selected model absent after normalization: %s',model);
    end

    baselineCounts = unique(rows.BaselineExact);
    sampleCounts = unique(rows.Samples);
    if numel(baselineCounts)~=1 || numel(sampleCounts)~=1
        error('%s has inconsistent baseline or sample counts.',model);
    end
    baselineExact = baselineCounts(1);
    nSamples = sampleCounts(1);
    baselinePct = 100*baselineExact/nSamples;
    allPercent(end+1) = baselinePct; %#ok<AGROW>

    xBase = im + offsets(1);
    scatter(ax,xBase,baselinePct,24,'x', ...
        'MarkerEdgeColor',[0.12 0.12 0.12],'LineWidth',1.0);
    text(ax,xBase,baselinePct+0.8,sprintf('%d/%d',baselineExact,nSamples), ...
        'HorizontalAlignment','center','VerticalAlignment','bottom', ...
        'FontName','Arial','FontSize',5.5,'Color',[0.12 0.12 0.12], ...
        'Margin',1.5);

    for ic = 1:3
        row = rows(rows.Condition==conditions(ic),:);
        if height(row)~=1
            error('%s requires exactly one %s endpoint row.',model,conditions(ic));
        end
        exact = row.SteeredExact;
        pct = 100*exact/nSamples;
        allPercent(end+1) = pct; %#ok<AGROW>
        x = im + offsets(ic+1);

        hConditions(ic) = scatter(ax,x,pct,24,markers{ic}, ...
            'MarkerEdgeColor',colors(ic,:), ...
            'MarkerFaceColor',colors(ic,:), ...
            'LineWidth',0.7);

        label = sprintf('%d/%d',exact,nSamples);
        if conditions(ic)=="shared"
            weight = 'bold';
            labelY = pct+1.2;
        else
            weight = 'normal';
            if conditions(ic)=="p_only"
                labelY = pct+2.2;
            else
                labelY = pct+0.8;
            end
        end
        text(ax,x,labelY,label, ...
            'Interpreter','tex', ...
            'HorizontalAlignment','center','VerticalAlignment','bottom', ...
            'FontName','Arial','FontSize',5.5,'FontWeight',weight, ...
            'Color',colors(ic,:),'Margin',1.5);
        if conditions(ic)=="shared"
            text(ax,x,pct-1.0,sprintf('\\Delta %+d',exact-baselineExact), ...
                'Interpreter','tex', ...
                'HorizontalAlignment','center','VerticalAlignment','top', ...
                'FontName','Arial','FontSize',5.5,'FontWeight','bold', ...
                'Color',colors(ic,:),'Margin',1.5);
        end
    end
end

if isempty(forcedYLim)
    yLimits = exact_ylim(allPercent);
else
    validateattributes(forcedYLim,{'numeric'},{'vector','numel',2,'finite'});
    yLimits = double(forcedYLim(:)');
end

xlim(ax,[0.45 nModels+0.55]);
ylim(ax,yLimits);
xticks(ax,1:nModels);
xticklabels(ax,repmat({''},1,nModels));
ySpan = diff(yLimits);
for im = 1:nModels
    text(ax,im,yLimits(1)-0.085*ySpan,display_names(selectedModels(im)), ...
        'HorizontalAlignment','center','VerticalAlignment','top', ...
        'FontName','Arial','FontSize',6,'Color','k', ...
        'Interpreter','none','Clipping','off');
end
ylabel(ax,'Exact transcription rate (%)','FontName','Arial','FontSize',6,'Color','k');
title(ax,"Maximum-dose steering (" + datasetTitle + ")", ...
    'FontName','Arial','FontSize',7,'FontWeight','bold','Color','k');

set(ax,'FontName','Arial','FontSize',7,'LineWidth',0.5, ...
    'XColor','k','YColor','k','TickDir','out','TickLength',[0.012 0.012], ...
    'Box','off','Color','w','XGrid','off','YGrid','off');

lgd = legend(ax,[hBaseline;hConditions], ...
    [{'Baseline'} conditionLabels], ...
    'Orientation','horizontal','Location','southoutside', ...
    'NumColumns',4,'Box','off','FontName','Arial','FontSize',5.5,'TextColor','k');
lgd.Units = 'normalized';
lgd.Position = [0.07 0.015 0.90 0.075];
lgd.ItemTokenSize = [7 4];
hold(ax,'off');
end

function yl = exact_ylim(values)
values = double(values(:));
values = values(isfinite(values));
if isempty(values)
    yl = [0 100];
    return;
end
lo = min(values);
hi = max(values);
span = max(hi-lo,8);
padBelow = max(3,0.15*span);
padAbove = max(7,0.25*span); % room for two-line Shared labels
lo = max(0,5*floor((lo-padBelow)/5));
hi = min(100,5*ceil((hi+padAbove)/5));
if hi-lo<20
    centre = (hi+lo)/2;
    lo = max(0,5*floor((centre-10)/5));
    hi = min(100,5*ceil((centre+10)/5));
end
yl = [lo hi];
end

function labels = display_names(modelKeys)
labels = strings(size(modelKeys));
for i = 1:numel(modelKeys)
    key = normalize_key(modelKeys(i));
    switch key
        case "baichuan_audio_base"
            labels(i) = sprintf('Baichuan-\nAudio-Base');
        case "baichuan_audio_instruct"
            labels(i) = sprintf('Baichuan-\nAudio-Instruct');
        case "qwen2_audio"
            labels(i) = sprintf('Qwen2-\nAudio');
        case "qwen2_audio_instruct"
            labels(i) = sprintf('Qwen2-\nAudio-Instruct');
        case "qwen_audio"
            labels(i) = sprintf('Qwen-\nAudio');
        case "qwen_audio_chat"
            labels(i) = sprintf('Qwen-\nAudio-Chat');
        case "covo_audio_chat"
            labels(i) = sprintf('Covo-\nAudio-Chat');
        case "kimi_audio_7b_instruct"
            labels(i) = sprintf('Kimi-Audio-\n7B-Instruct');
        otherwise
            labels(i) = strrep(key,'_','-');
    end
end
end

function print_endpoint_values(E,selectedModels)
conditions = ["p_only" "s_only" "shared"];
for i = 1:numel(selectedModels)
    model = normalize_key(selectedModels(i));
    rows = E(E.Model==model,:);
    base = unique(rows.BaselineExact);
    n = unique(rows.Samples);
    fprintf('  %-28s baseline %d/%d',model,base,n);
    for j = 1:numel(conditions)
        row = rows(rows.Condition==conditions(j),:);
        fprintf(' | %s %d/%d',conditions(j),row.SteeredExact,n);
    end
    fprintf('\n');
end
end

function name = choose_variable(names,candidates)
for i = 1:numel(candidates)
    idx = find(strcmpi(names,candidates(i)),1,'first');
    if ~isempty(idx)
        name = char(names(idx));
        return;
    end
end
error('Required column absent. Accepted names: %s',strjoin(candidates,', '));
end

function values = numeric_column(column,name)
if isnumeric(column) || islogical(column)
    values = double(column);
else
    values = str2double(string(column));
end
if any(isnan(values))
    error('Column %s contains non-numeric values.',name);
end
end

function value = normalize_key(value)
value = lower(strtrim(string(value)));
value = regexprep(value,'[^a-z0-9]+','_');
value = regexprep(value,'^_+|_+$','');
end

function value = normalize_condition(value)
value = normalize_key(value);
value(value=="ponly" | value=="p" | value=="phonological") = "p_only";
value(value=="sonly" | value=="s" | value=="semantic") = "s_only";
value(value=="p2s" | value=="shared_units") = "shared";
end

function export_figure(fig,outDir,name,savePDF,savePNG,resolution)
if savePDF
    exportgraphics(fig,fullfile(outDir,[name '.pdf']),'ContentType','vector');
end
if savePNG
    exportgraphics(fig,fullfile(outDir,[name '.png']),'Resolution',resolution);
end
end
