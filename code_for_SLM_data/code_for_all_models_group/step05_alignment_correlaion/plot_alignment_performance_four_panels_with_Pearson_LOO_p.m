%% Full-model alignment versus performance: Spearman correlations
% Four publication-style panels:
% 1) 14 models x 3 unit classes: sequential alignment
% 2) 14 models x 3 unit classes: hierarchy alignment
% 3) Shared sequential alignment vs mean performance (Spearman; rank plot)
% 4) Shared hierarchy alignment vs mean performance (Spearman; rank plot)
%
% Input files in the same folder:
%   combined_alignment_summary.xlsx
%   fleurs_meeting_dataset_asr_metrics.xlsx
%
% Output:
%   matlab_results/alignment_performance_four_panels/
%
% Mean performance = mean(Fleurs-zh, WenetSpeech meeting), both 1-CER.

clearvars; close all; clc;

%% Paths
here=fileparts(mfilename('fullpath'));
if isempty(here), here=pwd; end
alignmentFile=fullfile(here,'combined_alignment_summary.xlsx');
performanceFile=fullfile(here,'fleurs_meeting_dataset_asr_metrics.xlsx');
outDir=fullfile(here,'matlab_results','alignment_performance_four_panels');
if ~exist(outDir,'dir'), mkdir(outDir); end
assert(isfile(alignmentFile),'Cannot find %s',alignmentFile);
assert(isfile(performanceFile),'Cannot find %s',performanceFile);

%% Global style: Arial 6 pt, black text/axes, 0.5 pt axes
set(groot,'defaultAxesFontName','Arial','defaultTextFontName','Arial', ...
    'defaultAxesFontSize',6,'defaultTextFontSize',6, ...
    'defaultAxesLineWidth',0.5,'defaultAxesXColor','k','defaultAxesYColor','k', ...
    'defaultTextColor','k');

Pcolor=[180 37 35]/255;
Scolor=[34 91 142]/255;
SharedColor=[103 38 141]/255;
fitColor=[0 0 0];
ciColor=[0.82 0.82 0.82];

%% Read alignment
A=readtable(alignmentFile,'Sheet','AlignmentSummary', ...
    'TextType','string','VariableNamingRule','preserve');

%% Read performance
sheets=sheetnames(performanceFile);
if any(strcmp(sheets,'ASR_accuracy_1_minus_CER'))
    perfSheet='ASR_accuracy_1_minus_CER';
else
    perfSheet=sheets{1};
end
P=readtable(performanceFile,'Sheet',perfSheet, ...
    'TextType','string','VariableNamingRule','preserve');

requiredA=["Model","ModelSlug", ...
    "SequentialPearson16WithinChannelZ_P-only", ...
    "SequentialPearson16WithinChannelZ_S-only", ...
    "SequentialPearson16WithinChannelZ_Shared", ...
    "HierarchyPearson12_P-only", ...
    "HierarchyPearson12_S-only", ...
    "HierarchyPearson12_Shared"];
requiredP=["Model","ModelKey","Fleurs-zh","WenetSpeech meeting"];
assert(all(ismember(requiredA,string(A.Properties.VariableNames))), ...
    'combined_alignment_summary.xlsx is missing required columns.');
assert(all(ismember(requiredP,string(P.Properties.VariableNames))), ...
    'fleurs_meeting_dataset_asr_metrics.xlsx is missing required columns.');

%% Fixed model-key order only for matching
keyOrder=["baichuan_audio_base","baichuan_audio_instruct","covo_audio_chat", ...
    "firered_audio","freeze_omni","glm_4_voice","kimi_audio_7b_instruct", ...
    "minicpm_o_2_6","minicpm_o_4_5","qwen_audio","qwen_audio_chat", ...
    "qwen2_audio","qwen2_audio_instruct","qwen2_5_omni"];

A.Model=string(A.Model); A.ModelSlug=string(A.ModelSlug);
P.Model=string(P.Model); P.ModelKey=string(P.ModelKey);
A=sort_table_by_key(A,'ModelSlug',keyOrder);
P=sort_table_by_key(P,'ModelKey',keyOrder);
assert(height(A)==14 && height(P)==14,'Expected exactly 14 models in both files.');
assert(isequal(A.ModelSlug,P.ModelKey),'Model keys do not match between the two files.');

%% Extract metrics
seqP=double(A.("SequentialPearson16WithinChannelZ_P-only"));
seqS=double(A.("SequentialPearson16WithinChannelZ_S-only"));
seqShared=double(A.("SequentialPearson16WithinChannelZ_Shared"));
hierP=double(A.("HierarchyPearson12_P-only"));
hierS=double(A.("HierarchyPearson12_S-only"));
hierShared=double(A.("HierarchyPearson12_Shared"));

fleurs=double(P.("Fleurs-zh"));
meeting=double(P.("WenetSpeech meeting"));
meanPerformance=mean([fleurs meeting],2,'omitnan');

modelNames=string(A.Model);
shortNames=short_model_names(A.ModelSlug);

%% Performance order
% Higher mean ASR performance is shown at the TOP of panels 1-2.
[~,perfOrderDesc]=sort(meanPerformance,'descend');
modelNamesPerf=modelNames(perfOrderDesc);
shortNamesPerf=shortNames(perfOrderDesc);

% Performance rank for correlation plots: 1 = lowest, 14 = highest.
perfRank=average_rank(meanPerformance);
rankColors=performance_rank_colors(14);
pointColors=interp1(1:14,rankColors,perfRank,'linear');
rowColorsPerf=pointColors(perfOrderDesc,:);

%% Spearman correlations for all six tests + BH-FDR correction
seqMetrics=[seqP seqS seqShared];
hierMetrics=[hierP hierS hierShared];
allMetrics=[seqMetrics hierMetrics];

rAll=nan(1,6);
pAll=nan(1,6);
for k=1:6
    [rAll(k),pAll(k)]=spearman_corr(meanPerformance,allMetrics(:,k));
end
qAll=fdr_bh_adjust(pAll);

rSeq=rAll(3); pSeq=pAll(3); qSeq=qAll(3);
rHier=rAll(6); pHier=pAll(6); qHier=qAll(6);

fprintf('\nSix Spearman correlations (BH-FDR across all 6 tests):\n');
metricLabels=["Sequential Phonology units","Sequential Semantic units","Sequential Shared units", ...
    "Hierarchy Phonology units","Hierarchy Semantic units","Hierarchy Shared units"];
for k=1:6
    fprintf('%-22s r = % .4f, p = %.6g, q = %.6g\n', ...
        char(metricLabels(k)),rAll(k),pAll(k),qAll(k));
end

%% Leave-one-model-out validation for Spearman
nModels=numel(modelNames);
looSpearmanR=nan(nModels,6); looSpearmanP=nan(nModels,6); looSpearmanQ=nan(nModels,6);

for omit=1:nModels
    keep=true(nModels,1);
    keep(omit)=false;
    for k=1:6
        [looSpearmanR(omit,k),looSpearmanP(omit,k)]= ...
            spearman_corr(meanPerformance(keep),allMetrics(keep,k));
    end
    % Recompute BH-FDR across the six tests for each LOO sample.
    looSpearmanQ(omit,:)=fdr_bh_adjust(looSpearmanP(omit,:));
end

%% Save correlation + LOO statistics
fullStats=make_full_stats_table(metricLabels,"Spearman",rAll,pAll,qAll,nModels);

looStats=make_loo_stats_table(metricLabels,"Spearman",modelNames,A.ModelSlug, ...
    looSpearmanR,looSpearmanP,looSpearmanQ,nModels-1);

looSummary=make_loo_summary_table(metricLabels,"Spearman",rAll,looSpearmanR,looSpearmanP);

statsFile=fullfile(outDir,'correlation_and_LOO_statistics.xlsx');
if isfile(statsFile), delete(statsFile); end
writetable(fullStats,statsFile,'Sheet','FullSample');
writetable(looStats,statsFile,'Sheet','LeaveOneOut');
writetable(looSummary,statsFile,'Sheet','LOO_Summary');

%% Heatmap matrices, ordered from high to low mean performance
seqMat=seqMetrics(perfOrderDesc,:);
hierMat=hierMetrics(perfOrderDesc,:);

%% Figure 0: all six Spearman correlations
% Top row: Sequential P-only / S-only / Shared
% Bottom row: Hierarchy P-only / S-only / Shared
fig=figure('Color','w','Position',[100 100 700 410],'Renderer','painters');
tl=tiledlayout(fig,2,3,'TileSpacing','compact','Padding','compact');
tl.OuterPosition=[0.05 0.08 0.75 0.84];

titles=["Phonology units","Semantic units","Shared units","Phonology units","Semantic units","Shared units"];
for k=1:6
    ax=nexttile(tl,k);
    if k<=3
        if k==1, ylab='Sequential alignment index rank'; else, ylab=''; end
    else
        if k==4, ylab='Hierarchy alignment index rank'; else, ylab=''; end
    end
    plot_rank_correlation_panel(ax,meanPerformance,allMetrics(:,k),pointColors,fitColor,ciColor, ...
        titles(k),ylab,rAll(k),pAll(k),qAll(k),modelNames,false);
end
sgtitle(tl,'Fleurs-Meeting mean: alignment versus performance (Spearman)', ...
    'FontName','Arial','FontSize',6,'FontWeight','normal','Color','k');
add_model_legend(fig,modelNames,meanPerformance,pointColors);
export_both(fig,outDir,'Fig0_all_six_spearman_correlations');
close(fig);

%% Figure 0L: leave-one-model-out Spearman robustness
plot_loo_six_panel(metricLabels,modelNames,meanPerformance,pointColors, ...
    rAll,looSpearmanR,looSpearmanP,"Spearman",outDir, ...
    'Fig0L_LOO_spearman_correlations');

%% Figure 1: sequential alignment heatmap
fig=figure('Color','w','Position',[100 100 300 360],'Renderer','painters');
ax=axes(fig,'Position',[0.18 0.12 0.68 0.78]);
plot_alignment_heatmap(ax,seqMat,'Sequential alignment index',rowColorsPerf);
cb=colorbar(ax,'eastoutside');
style_colorbar(cb);
export_both(fig,outDir,'Fig1_sequential_alignment_14models');
close(fig);

%% Figure 2: hierarchy alignment heatmap
fig=figure('Color','w','Position',[100 100 300 360],'Renderer','painters');
ax=axes(fig,'Position',[0.18 0.12 0.68 0.78]);
plot_alignment_heatmap(ax,hierMat,'Hierarchy alignment index',rowColorsPerf);
cb=colorbar(ax,'eastoutside');
style_colorbar(cb);
export_both(fig,outDir,'Fig2_hierarchy_alignment_14models');
close(fig);

%% Figure 3: shared sequential alignment vs mean performance
fig=figure('Color','w','Position',[100 100 470 300],'Renderer','painters');
ax=axes(fig,'Position',[0.13 0.16 0.52 0.75]);
plot_rank_correlation_panel(ax,meanPerformance,seqShared,pointColors,fitColor,ciColor, ...
    'Shared units','Sequential alignment index rank',rSeq,pSeq,qSeq,modelNames,true);
export_both(fig,outDir,'Fig3_shared_sequential_vs_performance');
close(fig);

%% Figure 4: shared hierarchy alignment vs mean performance
fig=figure('Color','w','Position',[100 100 470 300],'Renderer','painters');
ax=axes(fig,'Position',[0.13 0.16 0.52 0.75]);
plot_rank_correlation_panel(ax,meanPerformance,hierShared,pointColors,fitColor,ciColor, ...
    'Shared units','Hierarchy alignment index rank',rHier,pHier,qHier,modelNames,true);
export_both(fig,outDir,'Fig4_shared_hierarchy_vs_performance');
close(fig);

%% Figure 5: combined 2 x 2, width : height ~= 2 : 3
% Row 1 = Sequential; Row 2 = Hierarchy.
% Both heatmaps show the same performance-sorted model labels.
% No colorbar is included here; use the standalone heatmap colorbar later.
fig=figure('Color','w','Position',[100 60 420 540],'Renderer','painters');

% ----------------------------- Row 1: Sequential -------------------------
ax1=axes(fig,'Position',[0.13 0.58 0.30 0.31]);
plot_alignment_heatmap(ax1,seqMat,'Sequential alignment index',rowColorsPerf);
panel_letter(ax1,'a');

ax2=axes(fig,'Position',[0.56 0.58 0.38 0.31]);
plot_rank_correlation_panel(ax2,meanPerformance,seqShared,pointColors,fitColor,ciColor, ...
    'Shared units','Sequential alignment index rank',rSeq,pSeq,qSeq,modelNames,false);
panel_letter(ax2,'b');

% ----------------------------- Row 2: Hierarchy --------------------------
ax3=axes(fig,'Position',[0.13 0.10 0.30 0.31]);
plot_alignment_heatmap(ax3,hierMat,'Hierarchy alignment index',rowColorsPerf);
panel_letter(ax3,'c');

ax4=axes(fig,'Position',[0.56 0.10 0.38 0.31]);
plot_rank_correlation_panel(ax4,meanPerformance,hierShared,pointColors,fitColor,ciColor, ...
    'Shared units','Hierarchy alignment index rank',rHier,pHier,qHier,modelNames,false);
panel_letter(ax4,'d');

export_both(fig,outDir,'Fig5_combined_2x2');
close(fig);

fprintf('\nSaved figures to:\n%s\n',outDir);

%% ============================ Functions =================================
function T=sort_table_by_key(T,keyVar,keyOrder)
keys=string(T.(keyVar));
[tf,rank]=ismember(keys,keyOrder);
if any(~tf), error('Unexpected model keys: %s',strjoin(keys(~tf),', ')); end
if numel(unique(keys))~=numel(keys), error('Duplicate model keys detected in %s.',keyVar); end
[~,ord]=sort(rank); T=T(ord,:);
end

function labels=short_model_names(slugs)
slugs=string(slugs); labels=strings(size(slugs));
for i=1:numel(slugs)
    switch slugs(i)
        case "baichuan_audio_base", labels(i)="Baichuan-Base";
        case "baichuan_audio_instruct", labels(i)="Baichuan-Inst";
        case "covo_audio_chat", labels(i)="Covo";
        case "firered_audio", labels(i)="FireRed";
        case "freeze_omni", labels(i)="Freeze";
        case "glm_4_voice", labels(i)="GLM-4";
        case "kimi_audio_7b_instruct", labels(i)="Kimi";
        case "minicpm_o_2_6", labels(i)="MiniCPM-2.6";
        case "minicpm_o_4_5", labels(i)="MiniCPM-4.5";
        case "qwen_audio", labels(i)="Qwen";
        case "qwen_audio_chat", labels(i)="Qwen-Chat";
        case "qwen2_audio", labels(i)="Qwen2";
        case "qwen2_audio_instruct", labels(i)="Qwen2-Inst";
        case "qwen2_5_omni", labels(i)="Qwen2.5";
        otherwise, labels(i)=slugs(i);
    end
end
end

function plot_alignment_heatmap(ax,M,ttl,rowColors)
% 14-model x 3-unit-class heatmap with colored row markers indicating model identity.
% Rows are already sorted from high to low mean performance.
imagesc(ax,M);
colormap(ax,blue_white_red(256));
caxis(ax,[-1 1]);
set(ax,'YDir','reverse'); % row 1 = highest-performance model at the top
n=size(M,1);

yticks(ax,1:n);
yticklabels(ax,[]);
xticks(ax,1:3);
xticklabels(ax,[]);
ax.XTickLabelRotation=0;
xlim(ax,[0.5 3.5]); ylim(ax,[0.5 n+0.5]);

% Draw colored row markers on the left; colors match the scatter panels.
hold(ax,'on');
scatter(ax,0.28*ones(n,1),(1:n)',30,rowColors,'filled', ...
    'MarkerEdgeColor','none','LineWidth',0.2,'Clipping','off');

% Draw the three category labels manually to avoid MATLAB auto-rotation.
labelY=n+0.92;
text(ax,1,labelY,sprintf('Phonological\nunits'), ...
    'HorizontalAlignment','center','VerticalAlignment','top', ...
    'FontName','Arial','FontSize',6,'FontWeight','normal', ...
    'Color','k','Interpreter','none','Rotation',0,'Clipping','off');
text(ax,2,labelY,sprintf('Semantic\nunits'), ...
    'HorizontalAlignment','center','VerticalAlignment','top', ...
    'FontName','Arial','FontSize',6,'FontWeight','normal', ...
    'Color','k','Interpreter','none','Rotation',0,'Clipping','off');
text(ax,3,labelY,sprintf('Shared\nunits'), ...
    'HorizontalAlignment','center','VerticalAlignment','top', ...
    'FontName','Arial','FontSize',6,'FontWeight','normal', ...
    'Color','k','Interpreter','none','Rotation',0,'Clipping','off');

% Cell values
for i=1:n
    for j=1:3
        value=M(i,j);
        if isfinite(value)
            if abs(value)>=0.60
                txtColor='w';
            else
                txtColor='k';
            end
            text(ax,j,i,sprintf('%.2f',value), ...
                'HorizontalAlignment','center','VerticalAlignment','middle', ...
                'FontName','Arial','FontSize',5.2,'FontWeight','normal','Color',txtColor);
        end
    end
end

% Thin white cell separators.
for x=1.5:1:2.5
    xline(ax,x,'Color','w','LineWidth',0.45,'HandleVisibility','off');
end
for y=1.5:1:(n-0.5)
    yline(ax,y,'Color','w','LineWidth',0.35,'HandleVisibility','off');
end

title(ax,ttl,'FontName','Arial','FontSize',6,'FontWeight','bold','Color','k');
set(ax,'FontName','Arial','FontSize',6,'FontWeight','normal', ...
    'LineWidth',0.5,'XColor','k','YColor','k','TickDir','out', ...
    'TickLength',[0 0],'Box','on');
hold(ax,'off');
end

function cmap=blue_white_red(n)
blue=[35 100 156]/255;
white=[1 1 1];
red=[180 37 35]/255;
half=floor(n/2);
n2=n-half;
cmap=[interp1([1 half],[blue;white],1:half); ...
      interp1([1 n2+1],[white;red],2:n2+1)];
end

function style_colorbar(cb)
cb.FontName='Arial';
cb.FontSize=6;
cb.Color='k';
cb.LineWidth=0.5;
cb.Ticks=[-1 0 1];
cb.Label.String='Alignment index';
cb.Label.FontName='Arial';
cb.Label.FontSize=6;
cb.Label.Color='k';
end

function plot_rank_correlation_panel(ax,performance,alignment,pointColors,fitColor,ciColor,ttl,ylab,rval,pval,qval,modelLabels,showLegend)
performance=double(performance(:)); alignment=double(alignment(:));
modelLabels=string(modelLabels(:));
good=isfinite(performance)&isfinite(alignment);
performance=performance(good);
alignment=alignment(good);
pointColors=pointColors(good,:);
modelLabels=modelLabels(good);
x=average_rank(performance);
y=average_rank(alignment);
hold(ax,'on');

% Linear fit on ranks only as a visual guide; Spearman r is the statistic.
[xx,yy,lo,hi]=rank_regression_band(x,y);
fill(ax,[xx fliplr(xx)],[lo fliplr(hi)],ciColor, ...
    'EdgeColor','none','FaceAlpha',0.60,'HandleVisibility','off');
plot(ax,xx,yy,'-','Color',fitColor,'LineWidth',0.9,'HandleVisibility','off');

% Draw each model separately so standalone plots can carry a model legend.
h=gobjects(numel(x),1);
for i=1:numel(x)
    h(i)=scatter(ax,x(i),y(i),30,'filled', ...
        'MarkerFaceColor',pointColors(i,:), ...
        'MarkerEdgeColor','none','LineWidth',0.35, ...
        'DisplayName',char(modelLabels(i)));
end

xlim(ax,[0.5 numel(x)+0.5]); ylim(ax,[0.5 numel(y)+0.5]);
xticks(ax,[1 4 8 12 14]); yticks(ax,[1 4 8 12 14]);
xlabel(ax,'Model performance rank','FontName','Arial','FontSize',6,'Color','k');
if strlength(string(ylab))>0
    ylabel(ax,ylab,'FontName','Arial','FontSize',6,'Color','k');
end
title(ax,ttl,'FontName','Arial','FontSize',6,'FontWeight','normal','Color','k');

stars=significance_stars(qval);
if qval<0.05
    statColor=[0.85 0 0];
else
    statColor='k';
end
statText=sprintf('\\itr\\rm = %.2f%s\n\\itp\\rm = %s, \\itq\\rm = %s', ...
    rval,stars,format_prob(pval),format_prob(qval));
text(ax,0.96,0.08,statText,'Units','normalized', ...
    'HorizontalAlignment','right','VerticalAlignment','bottom', ...
    'FontName','Arial','FontSize',6,'FontAngle','normal', ...
    'Interpreter','tex','Color',statColor);

if showLegend
    [~,legendOrder]=sort(x,'descend');
    lg=legend(ax,h(legendOrder),cellstr(modelLabels(legendOrder)), ...
        'Location','eastoutside','Box','off');
    set(lg,'FontName','Arial','FontSize',6,'TextColor','k');
end

style_axis(ax);
hold(ax,'off');
end

function plot_loo_six_panel(metricLabels,modelNames,performance,pointColors,fullR,looR,looP,corrType,outDir,fileBase)
n=numel(modelNames);
[~,ord]=sort(performance,'descend');

fig=figure('Color','w','Position',[100 100 700 410],'Renderer','painters');
tl=tiledlayout(fig,2,3,'TileSpacing','compact','Padding','compact');
tl.OuterPosition=[0.05 0.08 0.75 0.84];

for k=1:6
    ax=nexttile(tl,k);
    hold(ax,'on');

    yline(ax,0,':','Color',[0.65 0.65 0.65], ...
        'LineWidth',0.5,'HandleVisibility','off');
    yline(ax,fullR(k),'-','Color','k', ...
        'LineWidth',0.9,'HandleVisibility','off');

    % One point = correlation after omitting that model.
    % Red edge = uncorrected P < 0.05 for that leave-one-out correlation.
    for j=1:n
        i=ord(j);
        if isfinite(looP(i,k)) && looP(i,k)<0.05
            edgeColor=[0.85 0 0];
            edgeWidth=0.9;
        else
            edgeColor='none';
            edgeWidth=0.5;
        end
        scatter(ax,j,looR(i,k),24,'filled', ...
            'MarkerFaceColor',pointColors(i,:), ...
            'MarkerEdgeColor',edgeColor, ...
            'LineWidth',edgeWidth,'HandleVisibility','off');
    end

    xlim(ax,[0.5 n+0.5]);
    ylim(ax,[-1.05 1.05]);
    xticks(ax,[]);
    yticks(ax,[-1 -0.5 0 0.5 1]);

    if k<=3
        ttl=extractAfter(metricLabels(k),"Sequential ");
    else
        ttl=extractAfter(metricLabels(k),"Hierarchy ");
    end
    title(ax,ttl,'FontName','Arial','FontSize',6, ...
        'FontWeight','normal','Color','k');

    vals=looR(:,k);
    vals=vals(isfinite(vals));
    sigN=sum(isfinite(looP(:,k)) & looP(:,k)<0.05);
    validN=sum(isfinite(looP(:,k)));

    if ~isempty(vals)
        txt=sprintf('full \\itr\\rm = %.2f\nLOO %.2f to %.2f\n\\itp\\rm<0.05: %d/%d', ...
            fullR(k),min(vals),max(vals),sigN,validN);
        if sigN>0
            txtColor=[0.85 0 0];
        else
            txtColor='k';
        end
        text(ax,0.04,0.06,txt,'Units','normalized', ...
            'HorizontalAlignment','left','VerticalAlignment','bottom', ...
            'FontName','Arial','FontSize',5.5,'Interpreter','tex','Color',txtColor);
    end

    style_axis(ax);
    hold(ax,'off');
end

ylabel(tl,'Leave-one-model-out Spearman \itr','FontName','Arial','FontSize',6, ...
    'Interpreter','tex','Color','k');
xlabel(tl,'Omitted model (ordered by mean performance)', ...
    'FontName','Arial','FontSize',6,'Color','k');

sgtitle(tl,sprintf('Leave-one-model-out %s robustness',corrType), ...
    'FontName','Arial','FontSize',6,'FontWeight','normal','Color','k');

add_model_legend(fig,modelNames,performance,pointColors);
annotation(fig,'textbox',[0.81 0.08 0.18 0.06], ...
    'String','Red edge: \\itp\\rm < 0.05', ...
    'FontName','Arial','FontSize',6,'Interpreter','tex','Color',[0.85 0 0], ...
    'EdgeColor','none','HorizontalAlignment','left', ...
    'VerticalAlignment','middle');

export_both(fig,outDir,fileBase);
close(fig);
end

function [xx,yy,lo,hi]=linear_regression_band(x,y)
x=double(x(:)); y=double(y(:)); n=numel(x);
X=[ones(n,1) x];
b=X\y;
xx=linspace(min(x),max(x),200);
yy=b(1)+b(2)*xx;
res=y-X*b;
if n>2
    s2=sum(res.^2)/(n-2);
    xbar=mean(x);
    Sxx=sum((x-xbar).^2);
    se=sqrt(s2*(1/n+(xx-xbar).^2/max(Sxx,eps)));
    tcrit=2.0; % visual 95% band; correlation inference uses corr().
    lo=yy-tcrit*se;
    hi=yy+tcrit*se;
else
    lo=yy; hi=yy;
end
end

function T=make_full_stats_table(metricLabels,corrType,r,p,q,nModels)
T=table(metricLabels',repmat(corrType,6,1),repmat(nModels,6,1), ...
    r',p',q','VariableNames', ...
    {'Metric','CorrelationType','NModels','R','PValue','FDR_q'});
end

function T=make_loo_stats_table(metricLabels,corrType,modelNames,modelSlugs,R,P,Q,nUsed)
parts=cell(numel(modelNames),1);
for i=1:numel(modelNames)
    parts{i}=table( ...
        repmat(corrType,6,1), ...
        repmat(modelNames(i),6,1), ...
        repmat(modelSlugs(i),6,1), ...
        repmat(nUsed,6,1), ...
        metricLabels',R(i,:)',P(i,:)',Q(i,:)', ...
        'VariableNames',{'CorrelationType','OmittedModel','OmittedModelSlug', ...
        'NModelsUsed','Metric','R','PValue','FDR_q'});
end
T=vertcat(parts{:});
end

function T=make_loo_summary_table(metricLabels,corrType,fullR,looR,looP)
nMetric=numel(metricLabels);
meanR=nan(nMetric,1); minR=nan(nMetric,1); maxR=nan(nMetric,1);
sdR=nan(nMetric,1); sameSignN=zeros(nMetric,1); sigN=zeros(nMetric,1);
for k=1:nMetric
    vals=looR(:,k);
    good=isfinite(vals);
    if any(good)
        vv=vals(good);
        meanR(k)=mean(vv);
        minR(k)=min(vv);
        maxR(k)=max(vv);
        sdR(k)=std(vv,0);
        if fullR(k)==0
            sameSignN(k)=sum(vv==0);
        else
            sameSignN(k)=sum(sign(vv)==sign(fullR(k)));
        end
        sigN(k)=sum(isfinite(looP(:,k)) & looP(:,k)<0.05);
    end
end
T=table(metricLabels',repmat(corrType,nMetric,1),fullR', ...
    meanR,minR,maxR,sdR,sameSignN,sigN, ...
    'VariableNames',{'Metric','CorrelationType','FullSampleR', ...
    'LOO_MeanR','LOO_MinR','LOO_MaxR','LOO_SDR','SameSign_N','PSignificant_N'});
end

function [rho,p]=spearman_corr(x,y)
x=double(x(:)); y=double(y(:)); good=isfinite(x)&isfinite(y);
x=x(good); y=y(good);
if numel(x)<3 || std(x)==0 || std(y)==0
    rho=nan; p=nan; return;
end
[rho,p]=corr(x,y,'Type','Spearman');
end

function q=fdr_bh_adjust(p)
% Benjamini-Hochberg FDR-adjusted q values across the supplied tests.
p=double(p(:));
q=nan(size(p));
good=isfinite(p);
pg=p(good);
m=numel(pg);
if m==0, q=reshape(q,size(p)); return; end
[ps,ord]=sort(pg,'ascend');
qs=ps.*m./(1:m)';
for i=m-1:-1:1
    qs(i)=min(qs(i),qs(i+1));
end
qs=min(qs,1);
qg=nan(m,1);
qg(ord)=qs;
q(good)=qg;
q=reshape(q,size(p));
q=q(:)';
end

function stars=significance_stars(q)
if ~isfinite(q)
    stars='';
elseif q<0.001
    stars='***';
elseif q<0.01
    stars='**';
elseif q<0.05
    stars='*';
else
    stars='';
end
end

function add_model_legend(fig,modelLabels,performance,pointColors)
% One shared model legend for the six-panel correlation figure.
axL=axes(fig,'Position',[0.81 0.16 0.18 0.68],'Visible','off');
hold(axL,'on');
h=gobjects(numel(modelLabels),1);
for i=1:numel(modelLabels)
    h(i)=scatter(axL,nan,nan,30,'filled', ...
        'MarkerFaceColor',pointColors(i,:), ...
        'MarkerEdgeColor','none', ...
        'DisplayName',char(modelLabels(i)));
end
[~,ord]=sort(performance,'descend');
lg=legend(axL,h(ord),cellstr(string(modelLabels(ord))), ...
    'Location','west','Box','off');
set(lg,'FontName','Arial','FontSize',6,'TextColor','k');
hold(axL,'off');
end

function ranks=average_rank(x)
% Average ranks for ties; rank 1 = smallest value.
x=double(x(:)); ranks=nan(size(x));
good=isfinite(x); vals=x(good);
[sv,ord]=sort(vals,'ascend');
r=zeros(size(vals)); i=1;
while i<=numel(sv)
    j=i;
    while j<numel(sv) && sv(j+1)==sv(i), j=j+1; end
    r(ord(i:j))=mean(i:j);
    i=j+1;
end
ranks(good)=r;
end

function colors=performance_rank_colors(n)
% Yellow -> green -> blue -> purple, matching increasing performance rank.
anchors=[253 231 37; 94 201 98; 33 145 140; 59 82 139; 68 1 84]/255;
xa=linspace(1,n,size(anchors,1));
colors=interp1(xa,anchors,1:n,'linear');
end

function [xx,yy,lo,hi]=rank_regression_band(x,y)
x=double(x(:)); y=double(y(:)); n=numel(x);
X=[ones(n,1) x]; b=X\y;
xx=linspace(min(x),max(x),200);
yy=b(1)+b(2)*xx;
res=y-X*b;
if n>2
    s2=sum(res.^2)/(n-2);
    xbar=mean(x);
    Sxx=sum((x-xbar).^2);
    se=sqrt(s2*(1/n+(xx-xbar).^2/max(Sxx,eps)));
    tcrit=2.0; % close to 95% CI for n=14; visual guide only
    lo=yy-tcrit*se; hi=yy+tcrit*se;
else
    lo=yy; hi=yy;
end
end

function lim=nice_xlim(v,includeZero)
v=double(v(:)); v=v(isfinite(v));
mn=min(v); mx=max(v); span=max(mx-mn,0.10);
pad=0.08*span;
lim=[mn-pad mx+pad];
if includeZero && mn>=0, lim(1)=max(0,mn-pad); end
lim=0.05*[floor(lim(1)/0.05) ceil(lim(2)/0.05)];
if lim(1)==lim(2), lim=lim+[-0.05 0.05]; end
end

function s=format_prob(p)
if ~isfinite(p)
    s='NA';
elseif p<0.001
    s=sprintf('%.1e',p);
elseif p<0.1
    s=sprintf('%.4f',p);
else
    s=sprintf('%.3f',p);
end
end

function panel_letter(ax,letter)
text(ax,-0.17,1.08,letter,'Units','normalized', ...
    'FontName','Arial','FontSize',7,'FontWeight','bold','Color','k', ...
    'HorizontalAlignment','left','VerticalAlignment','top','Clipping','off');
end

function style_axis(ax)
set(ax,'FontName','Arial','FontSize',6,'FontWeight','normal', ...
    'LineWidth',0.5,'XColor','k','YColor','k','TickDir','out', ...
    'TickLength',[0.015 0.015],'Box','off');
ax.Title.Color='k'; ax.XLabel.Color='k'; ax.YLabel.Color='k';
end

function export_both(fig,outDir,baseName)
% Robust export: overwrite old files when possible; if a PNG is locked/open
% in another program, fall back to a new filename rather than stopping.
baseName=char(string(baseName));
if ~exist(outDir,'dir'), mkdir(outDir); end

pdfFile=fullfile(outDir,[baseName '.pdf']);
pngFile=fullfile(outDir,[baseName '.png']);

% PDF
if isfile(pdfFile)
    try
        delete(pdfFile);
    catch
        warning('Could not overwrite %s; it may be open in another program.',pdfFile);
    end
end
drawnow;
exportgraphics(fig,pdfFile,'ContentType','vector');

% PNG
if isfile(pngFile)
    try
        delete(pngFile);
    catch
        warning('Could not delete existing PNG (possibly open/locked): %s',pngFile);
    end
end

drawnow;
try
    exportgraphics(fig,pngFile,'Resolution',600);
catch ME
    warning('exportgraphics PNG failed: %s',ME.message);

    % First fallback: MATLAB print
    try
        print(fig,pngFile,'-dpng','-r600');
    catch
        % If the original PNG path is locked, save under a new filename.
        pngFile2=fullfile(outDir,[baseName '_new.png']);
        warning('Saving PNG with fallback filename: %s',pngFile2);
        if isfile(pngFile2)
            try, delete(pngFile2); catch, end
        end
        print(fig,pngFile2,'-dpng','-r600');
    end
end
end
