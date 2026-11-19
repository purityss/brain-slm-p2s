function step03_plot_model_sequential_shadedSEM(modelFilter)
% MATLAB sequential analysis for Human nominal + consecutive-2 model units.
% Heatmap display: ALL units are plotted (no row binning/downsampling).
% Heatmaps with >=1000 units use a ×10^3 y-axis exponent with compact ticks.
%
% OUTPUT:
% Figures: step03_sequential_hierarchy_geometry/matlab_results/01_sequential
% Excel:   step03_sequential_hierarchy_geometry/matlab_results/sequential_alignment.xlsx
%
% ================================ USAGE ==================================
% 1) Run ALL available models:
%       step03_plot_model_sequential_shadedSEM
%
% 2) Run ONE model only:
%       step03_plot_model_sequential_shadedSEM("qwen2_audio")
%
%    modelFilter must exactly match the model folder name (ModelSlug) under
%    the current classification directory, e.g. "qwen2_audio" rather than
%    the display/model name used in the paper.
%
% 3) Run several selected models:
%       step03_plot_model_sequential_shadedSEM(["qwen2_audio","qwen_audio"])
%
% IMPORTANT:
% - If modelFilter is omitted, all model folders containing
%   model_unit_classification_*.mat are processed.
% - Each run deletes and rewrites sequential_alignment.xlsx.
%   Therefore, running only one model will leave only that model in the
%   Excel output, although previously generated figure files are not deleted.
close all; clc;
if nargin<1, modelFilter=strings(0,1); else, modelFilter=string(modelFilter); end
thisFile = mfilename('fullpath');
classificationRoot = fileparts(thisFile);
groupRoot = fileparts(classificationRoot);
humanRoot = fullfile(groupRoot, 'step00_human_reference');
resultRoot = fullfile(groupRoot, 'step03_sequential_hierarchy_geometry');
figureRoot = fullfile(resultRoot, 'matlab_results', '01_sequential_shadedSEM');
excelFile = fullfile(resultRoot, 'matlab_results','sequential_alignment.xlsx');
if ~exist(figureRoot, 'dir'), mkdir(figureRoot); end
if exist(excelFile, 'file'), delete(excelFile); end
set(groot, 'defaultAxesFontName', 'Arial', 'defaultTextFontName', 'Arial', ...
    'defaultAxesFontSize', 6, 'defaultTextFontSize', 6, ...
    'defaultAxesLineWidth', 0.5, 'defaultAxesXColor', 'k', 'defaultAxesYColor', 'k');
canonical4 = [0.125 0.375 0.625 0.875];
centers16 = ((1:16) - 0.5) / 16;
classCodes = [1 2 3];
classNames = ["P-only" "S-only" "Shared"];
humanNames = ["Phonological" "Semantic" "Shared"];
pColor = [186 48 49] / 256;
sColor = [35 100 156] / 256;
heatmapZMax = 3;
% Match the human sequential main-figure display (step01, 0-800 ms).
showZeroSignificance = false; %true
clusterNperm = 5000; 
clusterFormingAlpha = 0.01;
sigAlpha = 0.01;
clusterSeed = 20260920;
rng(clusterSeed, 'twister');
humanTable = readtable(fullfile(humanRoot, 'Sequential_participant_level_statistics.csv'), ...
    'TextType', 'string', 'VariableNamingRule', 'preserve');
models = discover_models(classificationRoot);
if ~isempty(modelFilter), models=models(ismember(models.ModelSlug,modelFilter),:); end
models = sort_models_by_key(models);
if isempty(models), error('No requested model classification directories were found.'); end
alignmentParts = cell(0,1); trajectoryParts = cell(0,1);
for im = 1:height(models)
    slug = models.ModelSlug(im); modelName = models.Model(im);
    fprintf('[%d/%d] %s\n', im, height(models), modelName);
    modelDir = fullfile(classificationRoot, slug);
    matFile = one_file(modelDir, 'model_unit_classification_*.mat');
    layers = readtable(fullfile(modelDir, 'layer_metadata.csv'), ...
        'TextType', 'string', 'VariableNamingRule', 'preserve');
    layers = sortrows(layers, 'LayerOrderZeroBased');
    nUnits = sum(layers.Units);
    classCode = double(read_stat(matFile, 'class_code', nUnits));
    unitLayer = double(read_stat(matFile, 'layer_order_zero_based', nUnits));
    pz = double(read_stat(matFile, 'P_z', nUnits));
    sz = double(read_stat(matFile, 'S_z', nUnits));
    p4all = values_at_positions(pz, unitLayer, layers, canonical4);
    s4all = values_at_positions(sz, unitLayer, layers, canonical4);
    % Match the human sEEG sequential figure exactly in physical layout:
    % 315 x 260 px, 3 rows x 5 tiles, heatmap spans 2 tiles and line plot 3.
    fig = figure('Color','w','Position',[100 100 220 255],'Renderer','painters');
    tl = tiledlayout(fig,3,5,'TileSpacing','compact','Padding','compact');
    for ic = 1:3
        code = classCodes(ic); idx = find(classCode == code);
        p4 = p4all(idx,:); s4 = s4all(idx,:);
        tileHeat = (ic-1)*5 + 1;
        tileLine = (ic-1)*5 + 3;
        axH = nexttile(tl,tileHeat,[1 2]);
        axL = nexttile(tl,tileLine,[1 3]);
        if isempty(idx)
            axis(axH,'off');
            text(axH,.5,.5,"No " + classNames(ic) + " units", ...
                'HorizontalAlignment','center','FontName','Arial','FontSize',6,'Color','k');
            axis(axL,'off');
            pMean = nan(1,4); pSem = nan(1,4); sMean = nan(1,4); sSem = nan(1,4);
            contributing = []; pLayer = []; sLayer = [];
        else
            % --------------------- Combined heatmap -------------------------
            heat = sorted_heatmap(p4, s4, code);
            % Plot ALL classified units; do not bin/downsample heatmap rows.
            imagesc(axH,1:4,1:size(heat,1),heat);
            colormap(axH,blue_white_red(256));
            caxis(axH,[-heatmapZMax heatmapZMax]);
            title(axH,sprintf('%s model units (n = %s)', ...
                humanNames(ic),comma_int(numel(idx))), ...
                'FontName','Arial','FontSize',6,'FontWeight','normal','Color','k');
            ylabel(axH,{'Model unit','number'},'FontName','Arial','FontSize',6,'Color','k');
            xticks(axH,1:4);
            if ic==3
                xticklabels(axH,{'1','2','3','4'});
                xlabel(axH,'sequence segment','FontName','Arial','FontSize',6,'Color','k');
            else
                xticklabels(axH,[]);
            end
            style_heat_axis(axH);
            format_heatmap_unit_axis(axH,size(heat,1));
            % --------------------- Layer-first trajectories -----------------
            [pMean,pSem,sMean,sSem,contributing,pLayer,sLayer] = ...
                layer_first_summary(p4,s4,unitLayer(idx));
            hold(axL,'on');
            % SEM ribbons, matching the human sequential main figure.
            fill(axL,[1:4 fliplr(1:4)], ...
                [pMean+pSem fliplr(pMean-pSem)],pColor, ...
                'FaceAlpha',0.18,'EdgeColor','none','HandleVisibility','off');
            fill(axL,[1:4 fliplr(1:4)], ...
                [sMean+sSem fliplr(sMean-sSem)],sColor, ...
                'FaceAlpha',0.18,'EdgeColor','none','HandleVisibility','off');
            plot(axL,1:4,pMean,'-o', ...
                'Color',pColor,'MarkerEdgeColor',pColor,'MarkerFaceColor',pColor, ...
                'LineWidth',1.0,'MarkerSize',2.0,'DisplayName','Phonological');
            plot(axL,1:4,sMean,'-o', ...
                'Color',sColor,'MarkerEdgeColor',sColor,'MarkerFaceColor',sColor, ...
                'LineWidth',1.0,'MarkerSize',2.0,'DisplayName','Semantic');
            % xticks(axL,1:4);
            xlim(axL,[0.8 4.2]);
            xticks(axL,1:4);
            % xticklabels(axL,{'1','2','3','4'});
            % xlabel(axL,'sequence segment','FontName','Arial','FontSize',6,'Color','k');
            if ic==3
                xticklabels(axL,{'1','2','3','4'});
                xlabel(axL,'sequence segment','FontName','Arial','FontSize',6,'Color','k');
            else
                xticklabels(axL,[]);
            end
            ylabel(axL,{'Representation','index'},'FontName','Arial','FontSize',6,'Color','k');
            title(axL,sprintf('%s | %d layers',humanNames(ic),numel(contributing)), ...
                'FontName','Arial','FontSize',6,'FontWeight','normal','Color','k');
            ylim(axL,human_style_ylim(pMean,pSem,sMean,sSem));
            style_line_axis(axL);
            % Layer-wise sign-flip cluster permutation; descriptive within-model inference.
            if showZeroSignificance && size(pLayer,1) >= 2
                [pSig,sSig] = cluster_perm_joint_positive( ...
                    pLayer,sLayer,clusterNperm,clusterFormingAlpha,sigAlpha);
                yl = ylim(axL); yr = yl(2)-yl(1);
                yP_sig = yl(2)-0.105*yr;
                yS_sig = yl(2)-0.170*yr;
                if any(pSig)
                    plot_sig_clusters(axL,1:4,pSig,yP_sig,pColor,1.3);
                end
                if any(sSig)
                    plot_sig_clusters(axL,1:4,sSig,yS_sig,sColor,1.3);
                end
            end
            hold(axL,'off');
        end
        human = sortrows(humanTable(humanTable.ContactClass==humanNames(ic),:),'WindowCenter_ms');
        if height(human)~=16, error('Expected 16 human windows for %s.',humanNames(ic)); end
        hp16 = human.P_mean_z(:)'; hs16 = human.S_mean_z(:)';
        p16 = interp_clamped(canonical4,pMean,centers16);
        s16 = interp_clamped(canonical4,sMean,centers16);
        model16 = within_channel_z_vector(p16,s16);
        human16 = within_channel_z_vector(hp16,hs16);
        pearson16 = safe_corr(model16,human16,'Pearson');
        alignmentParts{end+1,1} = table(modelName,slug,code,classNames(ic),numel(idx),numel(contributing), ...
            pearson16,'VariableNames',{'Model','ModelSlug','ClassCode','UnitClass','NUnits', ...
            'NContributingLayers','SequentialPearson16WithinChannelZ'}); %#ok\<AGROW>
        trajectoryParts{end+1,1} = table(repmat(modelName,4,1),repmat(slug,4,1),repmat(code,4,1), ...
            repmat(classNames(ic),4,1),(1:4)',canonical4',pMean',pSem',sMean',sSem', ...
            'VariableNames',{'Model','ModelSlug','ClassCode','UnitClass','CanonicalWindow', ...
            'NormalizedCenter','PMeanZ','PSEMZ','SMeanZ','SSEMZ'}); %#ok\<AGROW>
    end
    title(tl,modelName + ": sequential representational dynamics", ...
        'FontName','Arial','FontSize',6,'FontWeight','bold','Color','k');
    exportgraphics(fig,fullfile(figureRoot,slug+'_sequential.pdf'),'ContentType','vector');
    exportgraphics(fig,fullfile(figureRoot,slug+'_sequential.png'),'Resolution',600);
    close(fig);
end
alignment = vertcat(alignmentParts{:}); trajectories = vertcat(trajectoryParts{:});
alignment = sort_output_by_model_key(alignment,{'ClassCode'});
trajectories = sort_output_by_model_key(trajectories,{'ClassCode','CanonicalWindow'});
writetable(alignment,excelFile,'Sheet','Alignment');
writetable(trajectories,excelFile,'Sheet','Trajectories');
writetable(humanTable,excelFile,'Sheet','HumanReference');
fprintf('Saved figures: %s\nSaved alignment: %s\n',figureRoot,excelFile);
end
function models = discover_models(root)
d = dir(root); d = d([d.isdir]); d = d(~ismember({d.name},{'.','..'})); rows = cell(0,2);
for i=1:numel(d)
    if ~isempty(dir(fullfile(root,d(i).name,'model_unit_classification_*.mat')))
        s = readtable(fullfile(root,d(i).name,'classification_summary.csv'),'TextType','string');
        rows(end+1,:) = {string(s.Model(1)),string(d(i).name)}; %#ok\<AGROW>
    end
end
models = cell2table(rows,'VariableNames',{'Model','ModelSlug'});
end
function models = sort_models_by_key(models)
% Fixed model-key order used consistently for plotting/processing and Excel output.
keyOrder = ["baichuan_audio_base","baichuan_audio_instruct","covo_audio_chat", ...
    "firered_audio","freeze_omni","glm_4_voice","kimi_audio_7b_instruct", ...
    "minicpm_o_2_6","minicpm_o_4_5","qwen_audio","qwen_audio_chat", ...
    "qwen2_audio","qwen2_audio_instruct","qwen2_5_omni"];
[tf,rank] = ismember(models.ModelSlug,keyOrder);
if any(~tf)
    extra = models.ModelSlug(~tf);
    [~,extraOrd] = sort(lower(extra));
    extraRank = numel(keyOrder) + (1:numel(extraOrd));
    tmp = zeros(nnz(~tf),1);
    tmp(extraOrd) = extraRank;
    rank(~tf) = tmp;
end
[~,ord] = sort(rank);
models = models(ord,:);
end
function T = sort_output_by_model_key(T,secondaryVars)
keyOrder = ["baichuan_audio_base","baichuan_audio_instruct","covo_audio_chat", ...
    "firered_audio","freeze_omni","glm_4_voice","kimi_audio_7b_instruct", ...
    "minicpm_o_2_6","minicpm_o_4_5","qwen_audio","qwen_audio_chat", ...
    "qwen2_audio","qwen2_audio_instruct","qwen2_5_omni"];
[tf,rank] = ismember(T.ModelSlug,keyOrder);
if any(~tf)
    extra = unique(T.ModelSlug(~tf),'stable');
    [~,extraOrd] = sort(lower(extra));
    extra = extra(extraOrd);
    for i=1:numel(extra)
        rank(T.ModelSlug==extra(i)) = numel(keyOrder)+i;
    end
end
T.ModelKeyOrder = rank;
T = sortrows(T,[{'ModelKeyOrder'},secondaryVars]);
T.ModelKeyOrder = [];
end
function file = one_file(folder,pattern)
d=dir(fullfile(folder,pattern)); if numel(d)~=1, error('Expected one %s in %s.',pattern,folder); end
file=fullfile(d(1).folder,d(1).name);
end
function x = read_stat(file,name,nUnits)
x=squeeze(h5read(file,"/unit_statistics/"+name));
if iscell(x), sx=string(x); if all(ismember(upper(sx),["TRUE","FALSE"]),'all'), x=strcmpi(sx,"TRUE"); else, x=str2double(sx); end, end
if isvector(x), x=x(:); elseif size(x,1)~=nUnits && size(x,2)==nUnits, x=x'; end
if size(x,1)~=nUnits, error('Unexpected %s shape in %s.',name,file); end
end
function out = values_at_positions(values,unitLayer,layers,target)
out=nan(size(values,1),numel(target));
for i=1:height(layers)
    order=double(layers.LayerOrderZeroBased(i)); use=unitLayer==order;
    nwin=double(layers.WindowCount(i)); edges=jsondecode(char(layers.WindowEdges(i)));
    centers=(double(edges(1:end-1))+double(edges(2:end)))/(2*double(layers.NativeSequenceLength(i)));
    block=values(use,1:nwin);
    indices=find(use);
    for j=1:numel(indices), out(indices(j),:)=interp_clamped(centers,block(j,:),target); end
end
end
function y = interp_clamped(x,v,xq)
x=double(x(:)'); v=double(v(:)'); good=isfinite(x)&isfinite(v);
if nnz(good)==0, y=nan(size(xq)); return; end
if nnz(good)==1, y=repmat(v(good),size(xq)); return; end
x=x(good); v=v(good); y=interp1(x,v,xq,'linear'); y(xq<=x(1))=v(1); y(xq>=x(end))=v(end);
end
function heat = sorted_heatmap(p,s,code)
pp=max(p,0); ss=max(s,0);
if code==1, score=sum(pp,2,'omitnan'); [~,ord]=sort(score,'descend');
elseif code==2, score=sum(ss,2,'omitnan'); [~,ord]=sort(score,'ascend');
else, score=sum(pp,2,'omitnan')-sum(ss,2,'omitnan'); [~,ord]=sort(score,'descend'); end
pp=pp(ord,:); ss=ss(ord,:); heat=zeros(size(pp)); up=pp>=ss & pp>0; us=ss>pp & ss>0;
heat(up)=pp(up); heat(us)=-ss(us);
end
function [pm,ps,sm,ss,layers,pl,sl] = layer_first_summary(p,s,unitLayer)
layers=unique(unitLayer(:)); pl=nan(numel(layers),4); sl=pl;
for i=1:numel(layers), use=unitLayer==layers(i); pl(i,:)=mean(p(use,:),1,'omitnan'); sl(i,:)=mean(s(use,:),1,'omitnan'); end
pm=mean(pl,1,'omitnan'); sm=mean(sl,1,'omitnan');
np=sum(isfinite(pl),1); ns=sum(isfinite(sl),1);
ps=std(pl,0,1,'omitnan')./sqrt(np); ss=std(sl,0,1,'omitnan')./sqrt(ns);
end
function r = safe_corr(x,y,kind)
good=isfinite(x)&isfinite(y); x=x(good); y=y(good);
if numel(x)<3 || std(x)==0 || std(y)==0, r=nan; return; end
if strcmpi(kind,'Spearman'), r=corr(x(:),y(:),'Type','Spearman'); else, r=corr(x(:),y(:),'Type','Pearson'); end
end
function v = within_channel_z_vector(p,s)
% Z-score P and S trajectories separately, then concatenate [z(P), z(S)].
zp = zscore_1d_population(p);
zs = zscore_1d_population(s);
v = [zp zs];
end
function z = zscore_1d_population(x)
% Match NumPy std(ddof=0): population standard deviation.
x = double(x(:)');
if isempty(x) || any(~isfinite(x))
    z = nan(size(x)); return;
end
scale = std(x,1);
if ~isfinite(scale) || scale<=0
    z = nan(size(x)); return;
end
z = (x-mean(x))/scale;
end
function cmap = blue_white_red(n)
b=[35 100 156]/256; w=[1 1 1]; r=[186 48 49]/256; h=floor(n/2);
n2=n-h; cmap=[interp1([1 h],[b;w],1:h); interp1([1 n2+1],[w;r],2:n2+1)];
end
function format_heatmap_unit_axis(ax,nUnits)
% Show the true unit count on the heatmap y-axis.
% For >=1000 units, MATLAB displays a x10^3 scale factor and compact ticks.
if nUnits<=0, return; end
ylim(ax,[0.5 nUnits+0.5]);
if nUnits>=1000
    ax.YAxis.Exponent = 3;
    nK = nUnits/1000;
    if nK<=2
        stepK = 0.5;
    elseif nK<=5
        stepK = 1;
    elseif nK<=10
        stepK = 2;
    elseif nK<=25
        stepK = 5;
    else
        stepK = 10;
    end
    ticksK = stepK:stepK:floor(nK/stepK)*stepK;
    if isempty(ticksK)
        ticksK = nK;
    end
    yticks(ax,ticksK*1000);
else
    ax.YAxis.Exponent = 0;
    % Keep a compact set of integer ticks for smaller unit counts.
    rawStep = nUnits/4;
    niceSteps = [1 2 5 10 20 50 100 200 500];
    step = niceSteps(find(niceSteps>=rawStep,1,'first'));
    if isempty(step), step=ceil(rawStep/500)*500; end
    ticks = step:step:floor(nUnits/step)*step;
    if isempty(ticks), ticks=nUnits; end
    yticks(ax,ticks);
end
end
function style_heat_axis(ax)
set(ax,'FontName','Arial','FontSize',6,'LineWidth',0.5, ...
    'XColor','k','YColor','k','TickDir','out','TickLength',[0 0],'Box','on');
end
function style_line_axis(ax)
set(ax,'FontName','Arial','FontSize',6,'LineWidth',0.5, ...
    'XColor','k','YColor','k','TickDir','out','Box','off');
end
function yl = human_style_ylim(pm,ps,sm,ss)
% Match the class-specific automatic y-limit rule used in the human figure.
vals=[pm-ps pm+ps sm-ss sm+ss];
vals=vals(isfinite(vals));
if isempty(vals)
    yl=[-0.15 1];
    return;
end
ymin=min(vals); ymax=max(vals); yrange=ymax-ymin;
if yrange<=eps, yrange=max(0.5,abs(ymax)); end
lower=min(-0.20,ymin-0.10*yrange);
upperRaw=ymax+0.22*yrange;
upper=ceil(upperRaw/0.5)*0.5;
if upper<=lower, upper=lower+0.5; end
yl=[lower upper];
end
function s=comma_int(n)
s=regexprep(sprintf('%d',n),'(?<=\d)(?=(\d{3})+$)',',');
end

function [P_corr,S_corr,P_sig,S_sig] = cluster_perm_joint_positive(Psub,Ssub,nperm,forming_alpha,cluster_alpha)
% Joint one-sided sign-flip cluster permutation across P and S trajectories.
% Rows are model layers; results describe layer-wise consistency within this model.
Psub=double(Psub); Ssub=double(Ssub); nwin=size(Psub,2);
P_corr=ones(1,nwin); S_corr=ones(1,nwin);
P_sig=false(1,nwin); S_sig=false(1,nwin);
if size(Psub,1)<2, return; end
[tP,pP]=one_sample_t_and_p(Psub,'right');
[tS,pS]=one_sample_t_and_p(Ssub,'right');
maskP=isfinite(pP)&pP<forming_alpha&tP>0;
maskS=isfinite(pS)&pS<forming_alpha&tS>0;
[clustersP,massP]=positive_clusters(tP,maskP);
[clustersS,massS]=positive_clusters(tS,maskS);
null_max=zeros(nperm,1); nsub=size(Psub,1);
for ip=1:nperm
    flips=2*(rand(nsub,1)>0.5)-1;
    [tPperm,pPperm]=one_sample_t_and_p(Psub.*flips,'right');
    [tSperm,pSperm]=one_sample_t_and_p(Ssub.*flips,'right');
    maskPperm=isfinite(pPperm)&pPperm<forming_alpha&tPperm>0;
    maskSperm=isfinite(pSperm)&pSperm<forming_alpha&tSperm>0;
    [~,massPperm]=positive_clusters(tPperm,maskPperm);
    [~,massSperm]=positive_clusters(tSperm,maskSperm);
    maxP=0; maxS=0;
    if ~isempty(massPperm), maxP=max(massPperm); end
    if ~isempty(massSperm), maxS=max(massSperm); end
    null_max(ip)=max(maxP,maxS);
end
for ic=1:numel(clustersP)
    cp=(1+nnz(null_max>=massP(ic)))/(nperm+1);
    P_corr(clustersP{ic})=cp;
end
for ic=1:numel(clustersS)
    cp=(1+nnz(null_max>=massS(ic)))/(nperm+1);
    S_corr(clustersS{ic})=cp;
end
P_sig=P_corr<cluster_alpha; S_sig=S_corr<cluster_alpha;
end

function [tval,pval] = one_sample_t_and_p(X,tail)
X=double(X); nwin=size(X,2); tval=nan(1,nwin); pval=nan(1,nwin);
for w=1:nwin
    x=X(:,w); x=x(isfinite(x)); n=numel(x);
    if n<2, continue; end
    mu=mean(x); sx=std(x,0);
    if sx<=eps
        if abs(mu)<=eps, t=0; elseif mu>0, t=Inf; else, t=-Inf; end
    else
        t=mu/(sx/sqrt(n));
    end
    tval(w)=t; df=n-1;
    switch lower(tail)
        case 'right', pval(w)=1-tcdf(t,df);
        case 'both', pval(w)=2*(1-tcdf(abs(t),df));
        otherwise, error('Unknown tail: %s',tail);
    end
end
end

function [clusters,masses] = positive_clusters(tval,mask)
clusters={}; masses=[]; runs=logical_runs(mask);
for ir=1:size(runs,1)
    idx=runs(ir,1):runs(ir,2); idx=idx(tval(idx)>0);
    if isempty(idx), continue; end
    clusters{end+1}=idx; %#ok<AGROW>
    masses(end+1)=sum(tval(idx)); %#ok<AGROW>
end
end

function runs = logical_runs(mask)
mask=logical(mask(:)'); d=diff([false mask false]);
starts=find(d==1); stops=find(d==-1)-1; runs=[starts(:) stops(:)];
end

function plot_sig_clusters(ax,x,sig_mask,y,color_value,line_width)
runs=logical_runs(sig_mask);
for ir=1:size(runs,1)
    idx=runs(ir,1):runs(ir,2);
    plot(ax,[x(idx(1)) x(idx(end))],[y y],'-', ...
        'Color',color_value,'LineWidth',line_width,'HandleVisibility','off');
    if numel(idx)==1
        plot(ax,x(idx),y,'o','Color',color_value,'MarkerFaceColor',color_value, ...
            'MarkerSize',2.5,'HandleVisibility','off');
    end
end
end
