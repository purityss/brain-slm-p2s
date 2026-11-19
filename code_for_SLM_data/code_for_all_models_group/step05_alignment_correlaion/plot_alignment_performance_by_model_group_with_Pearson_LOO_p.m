%% Subgroup alignment versus performance: Spearman correlations
% Subgroup brain-model alignment vs ASR performance analyses.
%
% Two predefined model groups:
%   1) Audio-Language Models (n=11)
%   2) Multimodal Omni Models (n=3)
%
% For EACH group and EACH of six alignment metrics:
%   Sequential: P-only / S-only / Shared
%   Hierarchy:  P-only / S-only / Shared
%
% This script adds:
%   (1) Spearman correlation
%   (2) Leave-one-model-out (LOO) robustness validation for Spearman
%
% Multiple-comparison handling:
%   - Full-sample Spearman: BH-FDR across the 6 Spearman tests within each group.
%   - LOO robustness significance is judged by the uncorrected P value
%     (P < 0.05) for each leave-one-model-out correlation.
%   - LOO q values are still calculated/saved for reference, but are NOT used
%     to define LOO significance in the figures or summary counts.
%
% IMPORTANT:
%   The Multimodal Omni group has only n=3. Leaving one model out leaves n=2,
%   so a meaningful LOO correlation validation cannot be performed. The script
%   keeps the full-sample n=3 Spearman results, but records LOO values
%   as NaN for this group and prints a warning.
%
% Input files in the same folder:
%   combined_alignment_summary.xlsx
%   fleurs_meeting_dataset_asr_metrics.xlsx
%
% Output:
%   matlab_results/alignment_performance_by_group/
%
% Saved figures:
%   *_six_spearman.pdf/png
%   *_LOO_spearman.pdf/png       (only when n>=4)
%
% Saved statistics:
%   subgroup_correlation_and_LOO_statistics.xlsx
%     Sheet FullSample
%     Sheet LeaveOneOut
%     Sheet LOO_Summary

clearvars; close all; clc;

%% Paths
here=fileparts(mfilename('fullpath'));
if isempty(here), here=pwd; end
alignmentFile=fullfile(here,'combined_alignment_summary.xlsx');
performanceFile=fullfile(here,'fleurs_meeting_dataset_asr_metrics.xlsx');
outDir=fullfile(here,'matlab_results','alignment_performance_by_group');
if ~exist(outDir,'dir'), mkdir(outDir); end
assert(isfile(alignmentFile),'Cannot find %s',alignmentFile);
assert(isfile(performanceFile),'Cannot find %s',performanceFile);

%% Style
set(groot,'defaultAxesFontName','Arial','defaultTextFontName','Arial', ...
    'defaultAxesFontSize',6,'defaultTextFontSize',6, ...
    'defaultAxesLineWidth',0.5,'defaultAxesXColor','k','defaultAxesYColor','k', ...
    'defaultTextColor','k');
fitColor=[0 0 0];
ciColor=[0.82 0.82 0.82];

%% Read data
A=readtable(alignmentFile,'Sheet','AlignmentSummary', ...
    'TextType','string','VariableNamingRule','preserve');

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

A.Model=string(A.Model);
A.ModelSlug=string(A.ModelSlug);
P.Model=string(P.Model);
P.ModelKey=string(P.ModelKey);

%% Match the 14 models by model key
allKeys=["baichuan_audio_base","baichuan_audio_instruct","covo_audio_chat", ...
    "firered_audio","freeze_omni","glm_4_voice","kimi_audio_7b_instruct", ...
    "minicpm_o_2_6","minicpm_o_4_5","qwen_audio","qwen_audio_chat", ...
    "qwen2_audio","qwen2_audio_instruct","qwen2_5_omni"];

A=sort_table_by_key(A,'ModelSlug',allKeys);
P=sort_table_by_key(P,'ModelKey',allKeys);
assert(height(A)==14 && height(P)==14,'Expected exactly 14 models in both files.');
assert(isequal(A.ModelSlug,P.ModelKey),'Model keys do not match between the two files.');

%% Extract six alignment metrics and mean performance
seqP=double(A.("SequentialPearson16WithinChannelZ_P-only"));
seqS=double(A.("SequentialPearson16WithinChannelZ_S-only"));
seqShared=double(A.("SequentialPearson16WithinChannelZ_Shared"));
hierP=double(A.("HierarchyPearson12_P-only"));
hierS=double(A.("HierarchyPearson12_S-only"));
hierShared=double(A.("HierarchyPearson12_Shared"));
allMetrics=[seqP seqS seqShared hierP hierS hierShared];

fleurs=double(P.("Fleurs-zh"));
meeting=double(P.("WenetSpeech meeting"));
meanPerformance=mean([fleurs meeting],2,'omitnan');

modelNames=string(A.Model);
modelKeys=string(A.ModelSlug);

metricNames=["Sequential P-only","Sequential S-only","Sequential Shared", ...
    "Hierarchy P-only","Hierarchy S-only","Hierarchy Shared"];
panelTitles=["Phonological units","Semantic units","Shared units", ...
    "Phonological units","Semantic units","Shared units"];

%% Keep the SAME model colors as the full 14-model figure
globalPerfRank=average_rank(meanPerformance); % 1=lowest, 14=highest
globalColorMap=performance_rank_colors(14);
globalPointColors=interp1(1:14,globalColorMap,globalPerfRank,'linear');

%% Predefined model groups
audioLanguageKeys=[ ...
    "baichuan_audio_base", ...
    "baichuan_audio_instruct", ...
    "covo_audio_chat", ...
    "firered_audio", ...
    "freeze_omni", ...
    "glm_4_voice", ...
    "kimi_audio_7b_instruct", ...
    "qwen_audio", ...
    "qwen_audio_chat", ...
    "qwen2_audio", ...
    "qwen2_audio_instruct"];

omniKeys=[ ...
    "minicpm_o_2_6", ...
    "minicpm_o_4_5", ...
    "qwen2_5_omni"];

assert(isempty(intersect(audioLanguageKeys,omniKeys)), ...
    'The two model groups overlap.');
assert(isequal(sort([audioLanguageKeys omniKeys]),sort(modelKeys')), ...
    'The two groups do not exactly cover the 14 active models.');

groupNames=["Audio-Language Models","Multimodal Omni Models"];
groupSlugs=["audio_language_models","multimodal_omni_models"];
groupKeys={audioLanguageKeys,omniKeys};

%% Containers for Excel output
fullParts=cell(0,1);
looParts=cell(0,1);
looSummaryParts=cell(0,1);

%% Main group loop
for g=1:2
    use=ismember(modelKeys,groupKeys{g});
    namesG=modelNames(use);
    keysG=modelKeys(use);
    perfG=meanPerformance(use);
    metricsG=allMetrics(use,:);
    colorsG=globalPointColors(use,:);
    nG=nnz(use);

    %% ---------------------- Full-sample correlations ---------------------
    rS=nan(1,6); pS=nan(1,6);

    for k=1:6
        [rS(k),pS(k)]=spearman_corr(perfG,metricsG(:,k));
    end

    qS=fdr_bh_adjust(pS);

    fprintf('\n============================================================\n');
    fprintf('%s (n = %d)\n',char(groupNames(g)),nG);
    fprintf('============================================================\n');

    fprintf('\nSpearman | BH-FDR across 6 tests\n');
    for k=1:6
        fprintf('%-22s r = % .4f, p = %.6g, q = %.6g\n', ...
            char(metricNames(k)),rS(k),pS(k),qS(k));
    end

    fullParts{end+1,1}=make_full_table(groupNames(g),nG,metricNames, ...
        "Spearman",rS,pS,qS);

    %% ---------------------- Spearman 2 x 3 figure -----------------------
    fig=figure('Color','w','Position',[100 100 760 390],'Renderer','painters');
    tl=tiledlayout(fig,2,3,'TileSpacing','compact','Padding','compact');
    tl.OuterPosition=[0.05 0.08 0.75 0.84];

    for k=1:6
        ax=nexttile(tl,k);
        if k==1
            ylab='Sequential alignment index rank';
        elseif k==4
            ylab='Hierarchy alignment index rank';
        else
            ylab='';
        end
        plot_spearman_panel(ax,perfG,metricsG(:,k),colorsG,fitColor,ciColor, ...
            panelTitles(k),ylab,rS(k),pS(k),qS(k));
    end

    sgtitle(tl,sprintf('%s: alignment versus performance (Spearman)',groupNames(g)), ...
        'FontName','Arial','FontSize',6,'FontWeight','normal','Color','k');
    add_model_legend(fig,namesG,perfG,colorsG);
    export_both(fig,outDir,['Fig_' char(groupSlugs(g)) '_six_spearman']);
    close(fig);

    %% ---------------------- Leave-one-model-out --------------------------
    % Each row = one omitted model; each column = one of the six metrics.
    looRS=nan(nG,6); looPS=nan(nG,6); looQS=nan(nG,6);

    if nG>=4
        for omit=1:nG
            keep=true(nG,1);
            keep(omit)=false;

            for k=1:6
                [looRS(omit,k),looPS(omit,k)]= ...
                    spearman_corr(perfG(keep),metricsG(keep,k));
            end

            % FDR is recomputed within each LOO dataset.
            looQS(omit,:)=fdr_bh_adjust(looPS(omit,:));
        end

        % Full LOO tables
        looParts{end+1,1}=make_loo_table(groupNames(g),keysG,namesG,metricNames, ...
            "Spearman",looRS,looPS,looQS,nG-1,"");
        % Summary tables
        looSummaryParts{end+1,1}=make_loo_summary(groupNames(g),metricNames, ...
            "Spearman",rS,looRS,looPS);
        % LOO figures
        plot_loo_figure(groupNames(g),groupSlugs(g),metricNames, ...
            namesG,perfG,colorsG,rS,looRS,looPS,"Spearman",outDir);
    else
        warning('%s has n=%d. LOO would leave only n=%d, so LOO correlation validation is not estimable.', ...
            groupNames(g),nG,nG-1);

        note="LOO not estimable: removing one model leaves only n=2.";
        looParts{end+1,1}=make_loo_table(groupNames(g),keysG,namesG,metricNames, ...
            "Spearman",looRS,looPS,looQS,nG-1,note);
        looSummaryParts{end+1,1}=make_loo_summary(groupNames(g),metricNames, ...
            "Spearman",rS,looRS,looPS);
    end
end

%% Save Excel statistics
fullTable=vertcat(fullParts{:});
looTable=vertcat(looParts{:});
looSummary=vertcat(looSummaryParts{:});

statsFile=fullfile(outDir,'subgroup_correlation_and_LOO_statistics.xlsx');
if isfile(statsFile), delete(statsFile); end
writetable(fullTable,statsFile,'Sheet','FullSample');
writetable(looTable,statsFile,'Sheet','LeaveOneOut');
writetable(looSummary,statsFile,'Sheet','LOO_Summary');

fprintf('\nSaved figures and statistics to:\n%s\n',outDir);

%% ============================ Functions =================================
function T=sort_table_by_key(T,keyVar,keyOrder)
keys=string(T.(keyVar));
[tf,rank]=ismember(keys,keyOrder);
if any(~tf), error('Unexpected model keys: %s',strjoin(keys(~tf),', ')); end
if numel(unique(keys))~=numel(keys), error('Duplicate model keys detected in %s.',keyVar); end
[~,ord]=sort(rank);
T=T(ord,:);
end

function T=make_full_table(groupName,nG,metricNames,corrType,r,p,q)
T=table(repmat(groupName,6,1),repmat(nG,6,1),metricNames', ...
    repmat(corrType,6,1),r',p',q', ...
    'VariableNames',{'ModelGroup','NModels','Metric','CorrelationType', ...
    'R','PValue','FDR_q'});
end

function T=make_loo_table(groupName,keys,names,metricNames,corrType,R,P,Q,nUsed,note)
nModel=numel(names);
parts=cell(nModel,1);
for i=1:nModel
    parts{i}=table( ...
        repmat(groupName,6,1), ...
        repmat(corrType,6,1), ...
        repmat(keys(i),6,1), ...
        repmat(names(i),6,1), ...
        repmat(nUsed,6,1), ...
        metricNames',R(i,:)',P(i,:)',Q(i,:)', ...
        repmat(string(note),6,1), ...
        'VariableNames',{'ModelGroup','CorrelationType','OmittedModelKey', ...
        'OmittedModel','NModelsUsed','Metric','R','PValue','FDR_q','Note'});
end
T=vertcat(parts{:});
end

function T=make_loo_summary(groupName,metricNames,corrType,fullR,looR,looP)
nMetric=numel(metricNames);
meanR=nan(nMetric,1);
minR=nan(nMetric,1);
maxR=nan(nMetric,1);
sdR=nan(nMetric,1);
validN=zeros(nMetric,1);
sameSignN=zeros(nMetric,1);
sigPN=zeros(nMetric,1);

for k=1:nMetric
    vals=looR(:,k);
    good=isfinite(vals);
    validN(k)=nnz(good);

    if validN(k)>0
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

        sigPN(k)=sum(isfinite(looP(:,k)) & looP(:,k)<0.05);
    end
end

T=table(repmat(groupName,nMetric,1),repmat(corrType,nMetric,1),metricNames', ...
    fullR',meanR,minR,maxR,sdR,validN,sameSignN,sigPN, ...
    'VariableNames',{'ModelGroup','CorrelationType','Metric','FullSampleR', ...
    'LOO_MeanR','LOO_MinR','LOO_MaxR','LOO_SDR','ValidLOO_N', ...
    'SameSign_N','PSignificant_N'});
end

function plot_spearman_panel(ax,performance,alignment,pointColors,fitColor,ciColor,ttl,ylab,rval,pval,qval)
performance=double(performance(:));
alignment=double(alignment(:));
good=isfinite(performance)&isfinite(alignment);
performance=performance(good);
alignment=alignment(good);
pointColors=pointColors(good,:);

x=average_rank(performance);
y=average_rank(alignment);
n=numel(x);

hold(ax,'on');
if n>=3 && range(x)>0
    [xx,yy,lo,hi]=linear_regression_band(x,y);
    fill(ax,[xx fliplr(xx)],[lo fliplr(hi)],ciColor, ...
        'EdgeColor','none','FaceAlpha',0.60,'HandleVisibility','off');
    plot(ax,xx,yy,'-','Color',fitColor,'LineWidth',0.9,'HandleVisibility','off');
end
for i=1:n
    scatter(ax,x(i),y(i),22,'filled', ...
        'MarkerFaceColor',pointColors(i,:), ...
        'MarkerEdgeColor','none','HandleVisibility','off');
end

xlim(ax,[0.5 n+0.5]);
ylim(ax,[0.5 n+0.5]);
set_rank_ticks(ax,n);
xlabel(ax,'Model performance rank','FontName','Arial','FontSize',6,'Color','k');
if strlength(string(ylab))>0
    ylabel(ax,ylab,'FontName','Arial','FontSize',6,'Color','k');
end
title(ax,ttl,'FontName','Arial','FontSize',6,'FontWeight','normal','Color','k');
add_stat_text(ax,rval,pval,qval);
style_axis(ax);
hold(ax,'off');
end

function add_stat_text(ax,rval,pval,qval)
stars=significance_stars(qval);
if isfinite(qval) && qval<0.05
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
end

function plot_loo_figure(groupName,groupSlug,metricNames,modelNames,performance,pointColors,fullR,looR,looP,corrType,outDir)
n=numel(modelNames);
[~,ord]=sort(performance,'descend');

fig=figure('Color','w','Position',[100 100 760 390],'Renderer','painters');
tl=tiledlayout(fig,2,3,'TileSpacing','compact','Padding','compact');
tl.OuterPosition=[0.05 0.08 0.75 0.84];

for k=1:6
    ax=nexttile(tl,k);
    hold(ax,'on');

    yline(ax,0,':','Color',[0.65 0.65 0.65],'LineWidth',0.5,'HandleVisibility','off');
    yline(ax,fullR(k),'-','Color','k','LineWidth',0.9,'HandleVisibility','off');

    for j=1:n
        i=ord(j);

        % All LOO points retain the model color.
        % Significant iterations (uncorrected P<0.05) are highlighted with a red edge.
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
            'LineWidth',edgeWidth, ...
            'HandleVisibility','off');
    end

    xlim(ax,[0.5 n+0.5]);
    ylim(ax,[-1.05 1.05]);
    xticks(ax,[]);
    yticks(ax,[-1 -0.5 0 0.5 1]);

    unitClassTitles=["Phonological units","Semantic units","Shared units"];
    titleClass=unitClassTitles(mod(k-1,3)+1);
    title(ax,titleClass,'FontName','Arial','FontSize',6,'FontWeight','normal','Color','k');

    vals=looR(:,k);
    vals=vals(isfinite(vals));
    sigN=sum(isfinite(looP(:,k)) & looP(:,k)<0.05);
    validN=sum(isfinite(looP(:,k)));

    if ~isempty(vals)
        txt=sprintf('full \\itr\\rm = %.2f\nLOO %.2f to %.2f\n\\itp\\rm < 0.05: %d/%d', ...
            fullR(k),min(vals),max(vals),sigN,validN);
        if sigN>0
            txtColor=[0.85 0 0];
        else
            txtColor='k';
        end
        text(ax,0.04,0.06,txt,'Units','normalized', ...
            'HorizontalAlignment','left','VerticalAlignment','bottom', ...
            'FontName','Arial','FontSize',5.5,'FontAngle','normal', ...
            'Interpreter','tex','Color',txtColor);
    end

    style_axis(ax);
    hold(ax,'off');
end

ylabel(tl,'Leave-one-model-out Spearman \itr','FontName','Arial','FontSize',6, ...
    'Interpreter','tex','Color','k');
xlabel(tl,'Omitted model (ordered by mean performance)','FontName','Arial','FontSize',6,'Color','k');

sgtitle(tl,sprintf('%s: leave-one-model-out %s robustness',groupName,corrType), ...
    'FontName','Arial','FontSize',6,'FontWeight','normal','Color','k');

% Shared model legend.
add_model_legend(fig,modelNames,performance,pointColors);

% Significance key for LOO points.
annotation(fig,'textbox',[0.81 0.08 0.18 0.06], ...
    'String','Red edge: \itp\rm < 0.05', ...
    'FontName','Arial','FontSize',6,'Interpreter','tex','Color',[0.85 0 0], ...
    'EdgeColor','none','HorizontalAlignment','left','VerticalAlignment','middle');

export_both(fig,outDir,['Fig_' char(groupSlug) '_LOO_' lower(char(corrType))]);
close(fig);
end

function [r,p]=spearman_corr(x,y)
x=double(x(:)); y=double(y(:));
good=isfinite(x)&isfinite(y);
x=x(good); y=y(good);
if numel(x)<3 || std(x)==0 || std(y)==0
    r=nan; p=nan; return;
end
[r,p]=corr(x,y,'Type','Spearman');
end

function ranks=average_rank(x)
x=double(x(:));
ranks=nan(size(x));
good=isfinite(x);
vals=x(good);
[sv,ord]=sort(vals,'ascend');
r=zeros(size(vals));
i=1;
while i<=numel(sv)
    j=i;
    while j<numel(sv) && sv(j+1)==sv(i), j=j+1; end
    r(ord(i:j))=mean(i:j);
    i=j+1;
end
ranks(good)=r;
end

function q=fdr_bh_adjust(p)
p=double(p(:));
q=nan(size(p));
good=isfinite(p);
pg=p(good);
m=numel(pg);
if m==0, q=q'; return; end
[ps,ord]=sort(pg,'ascend');
qs=ps.*m./(1:m)';
for i=m-1:-1:1
    qs(i)=min(qs(i),qs(i+1));
end
qs=min(qs,1);
qg=nan(m,1);
qg(ord)=qs;
q(good)=qg;
q=q';
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
    tcrit=approx_tcrit_95(n-2);
    lo=yy-tcrit*se;
    hi=yy+tcrit*se;
else
    lo=yy;
    hi=yy;
end
end

function t=approx_tcrit_95(df)
if df<=1
    t=12.706;
elseif df==2
    t=4.303;
elseif df==3
    t=3.182;
elseif df==4
    t=2.776;
elseif df==5
    t=2.571;
elseif df==6
    t=2.447;
elseif df==7
    t=2.365;
elseif df==8
    t=2.306;
elseif df==9
    t=2.262;
elseif df==10
    t=2.228;
elseif df<=12
    t=2.18;
else
    t=2.00;
end
end

function set_rank_ticks(ax,n)
if n<=4
    ticks=1:n;
else
    ticks=unique([1 4 8 12 n]);
    ticks=ticks(ticks<=n);
end
xticks(ax,ticks);
yticks(ax,ticks);
end

function colors=performance_rank_colors(n)
anchors=[253 231 37;94 201 98;33 145 140;59 82 139;68 1 84]/255;
xa=linspace(1,n,size(anchors,1));
colors=interp1(xa,anchors,1:n,'linear');
end

function add_model_legend(fig,modelLabels,performance,pointColors)
axL=axes(fig,'Position',[0.81 0.16 0.18 0.68],'Visible','off');
hold(axL,'on');
h=gobjects(numel(modelLabels),1);
for i=1:numel(modelLabels)
    h(i)=scatter(axL,nan,nan,18,'filled', ...
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

function style_axis(ax)
set(ax,'FontName','Arial','FontSize',6,'FontWeight','normal', ...
    'LineWidth',0.5,'XColor','k','YColor','k','TickDir','out', ...
    'TickLength',[0.015 0.015],'Box','off');
ax.Title.Color='k';
ax.XLabel.Color='k';
ax.YLabel.Color='k';
end

function export_both(fig,outDir,baseName)
baseName=char(string(baseName));
if ~exist(outDir,'dir'), mkdir(outDir); end
pdfFile=fullfile(outDir,[baseName '.pdf']);
pngFile=fullfile(outDir,[baseName '.png']);

if isfile(pdfFile)
    try, delete(pdfFile); catch, end
end
if isfile(pngFile)
    try, delete(pngFile); catch, end
end

drawnow;
exportgraphics(fig,pdfFile,'ContentType','vector');
drawnow;
try
    exportgraphics(fig,pngFile,'Resolution',600);
catch
    print(fig,pngFile,'-dpng','-r600');
end
end
