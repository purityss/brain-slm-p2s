function step04_plot_model_hierarchy(modelFilter)
% MATLAB hierarchy analysis using TotalUnits as the layer denominator.
% Correlations use slm_plot_safe_corr.m and require only base MATLAB.
% Figures: step03_sequential_hierarchy_geometry/matlab_results/02_hierarchy
% Excel:   step03_sequential_hierarchy_geometry/matlab_results/hierarchical_alignment.xlsx
%
% ================================ USAGE ==================================
% Run all available models:
%   step04_plot_model_hierarchy
% Run one model only:
%   step04_plot_model_hierarchy("qwen2_audio")
% Run several selected models:
%   step04_plot_model_hierarchy(["qwen2_audio","qwen_audio"])
%
% =========================== ARCHITECTURE RULE ============================
% IMPORTANT:
% CNN / audio encoder / adapter-projector / decoder boundaries are NOT
% hard-coded by layer number. They are determined separately for each model
% from layer_metadata.csv:
%   1) use Module / ModuleName / ArchitectureStage column if present;
%   2) otherwise infer the stage from LayerName + SequenceType.
% Each extracted layer keeps one equal step on the x-axis. Vertical dotted
% lines mark the actual module boundaries of that model. Smoothing is done
% WITHIN each architecture stage only, so CNN/encoder/decoder values are not
% mixed across module boundaries.
% X-axis tick labels are architecture-aware:
%   CNN end | Audio encoder end | Decoder start (when a bridge/projector exists) | Decoder end.
% If encoder connects directly to decoder, only one shared interface label is shown.
% The encoder-decoder bridge is visually expanded so nearby boundary labels do not overlap.
%
% Plot transforms affect visualization only. Hierarchy correlations always
% use the original unsmoothed layer proportions.
if nargin<1, modelFilter=strings(0,1); else, modelFilter=string(modelFilter); end
thisFile=mfilename('fullpath'); classificationRoot=fileparts(thisFile);
groupRoot=fileparts(classificationRoot); humanRoot=fullfile(groupRoot,'step00_human_reference');
resultRoot=fullfile(groupRoot,'step03_sequential_hierarchy_geometry');
figureRoot=fullfile(resultRoot,'matlab_results','02_hierarchy');
excelFile=fullfile(resultRoot,'matlab_results','hierarchical_alignment.xlsx');
if ~exist(figureRoot,'dir'), mkdir(figureRoot); end
set(groot,'defaultAxesFontName','Arial','defaultTextFontName','Arial', ...
    'defaultAxesFontSize',6,'defaultTextFontSize',6,'defaultAxesLineWidth',0.5, ...
    'defaultAxesXColor','k','defaultAxesYColor','k');
classCodes=[1 2 3]; classNames=["P-only" "S-only" "Shared"];
colors=[180 37 35;34 91 142;103 38 141]/255;
smoothWidth=5;
yBreaksOrig=[0 3 20 50 100];
yBreaksVis=[0 25 50 75 100];
yTickVals=[3 20 50 100];
human=readtable(fullfile(humanRoot,'Hierarchy_12ROI_distanceMedian_summary.csv'), ...
    'TextType','string','VariableNamingRule','preserve');
human=sortrows(human,'HierarchyRank');
humanCurves={human.Pct_Ponly(:),human.Pct_Sonly(:),human.Pct_Shared(:)};
models=discover_models(classificationRoot); layerParts=cell(0,1); alignParts=cell(0,1);
if ~isempty(modelFilter)
    missing=setdiff(modelFilter(:),string(models.ModelSlug),'stable');
    if ~isempty(missing), error('Missing classification MAT for requested model(s): %s',strjoin(missing,', ')); end
    models=models(ismember(models.ModelSlug,modelFilter),:);
end
models=sort_models_by_key(models);
if isempty(models), error('No requested model classification directories were found.'); end
modelFrames=cell(height(models),1);
for im=1:height(models)
    slug=models.ModelSlug(im); modelName=models.Model(im);
    fprintf('[%d/%d] %s\n',im,height(models),modelName);
    modelDir=fullfile(classificationRoot,slug);
    matFile=one_file(modelDir,'model_unit_classification_*.mat');
    layers=readtable(fullfile(modelDir,'layer_metadata.csv'),'TextType','string','VariableNamingRule','preserve');
    layers=sortrows(layers,'LayerOrderZeroBased');
    layers.ArchitectureStage=infer_architecture_stage(layers);
    print_architecture_summary(modelName,layers.ArchitectureStage);
    nUnits=sum(layers.Units);
    classCode=double(read_stat(matFile,'class_code',nUnits));
    eligible=logical(read_stat(matFile,'eligible',nUnits));
    unitLayer=double(read_stat(matFile,'layer_order_zero_based',nUnits));
    nL=height(layers); total=zeros(nL,1); elig=zeros(nL,1); inelig=zeros(nL,1);
    counts=zeros(nL,3); pct=zeros(nL,3);
    for il=1:nL
        order=double(layers.LayerOrderZeroBased(il)); use=unitLayer==order;
        total(il)=nnz(use); elig(il)=nnz(use & eligible); inelig(il)=total(il)-elig(il);
        if total(il)~=double(layers.Units(il)), error('%s layer %d unit-count mismatch.',modelName,order); end
        for ic=1:3
            counts(il,ic)=nnz(use & classCode==classCodes(ic));
            pct(il,ic)=100*counts(il,ic)/total(il);
        end
    end
    frame=table(repmat(modelName,nL,1),repmat(slug,nL,1),double(layers.LayerOrderZeroBased), ...
        double(layers.SourceLayerIndexZeroBased),layers.LayerName,layers.SequenceType,layers.ArchitectureStage, ...
        total,elig,inelig,counts(:,1),pct(:,1),counts(:,2),pct(:,2),counts(:,3),pct(:,3),linspace(0,1,nL)', ...
        'VariableNames',{'Model','ModelSlug','LayerOrderZeroBased','SourceLayerIndexZeroBased', ...
        'LayerName','SequenceType','ArchitectureStage','TotalUnits','EligibleUnits','IneligibleUnits', ...
        'N_Ponly','Pct_Ponly','N_Sonly','Pct_Sonly','N_Shared','Pct_Shared','NormalizedLayerDepth'});
    layerParts{end+1,1}=frame; modelFrames{im}=frame; %#ok<AGROW>
    for ic=1:3
        curve12=depth_bin(pct(:,ic),12); h12=humanCurves{ic};
        alignParts{end+1,1}=table(modelName,slug,classCodes(ic),classNames(ic),nL, ...
            safe_corr(curve12,h12,'Pearson'),safe_corr(curve12,h12,'Spearman'), ...
            'VariableNames',{'Model','ModelSlug','ClassCode','UnitClass','NLayers', ...
            'HierarchyPearson12','HierarchySpearman12'}); %#ok<AGROW>
    end
    pctSmooth=stagewise_smooth(pct,layers.ArchitectureStage,smoothWidth);
    fig=figure('Color','w','Position',[100 100 420 250],'Renderer','painters');
    ax=axes(fig,'Position',[0.16 0.18 0.80 0.66]); hold(ax,'on');
    xPlot=architecture_x_positions(layers.ArchitectureStage);
    plot(ax,xPlot,transform_prop_axis(pctSmooth(:,1),yBreaksOrig,yBreaksVis),'-','LineWidth',1.5,'Color',colors(1,:));
    plot(ax,xPlot,transform_prop_axis(pctSmooth(:,2),yBreaksOrig,yBreaksVis),'-','LineWidth',1.5,'Color',colors(2,:));
    plot(ax,xPlot,transform_prop_axis(pctSmooth(:,3),yBreaksOrig,yBreaksVis),'-','LineWidth',1.5,'Color',colors(3,:));
    add_architecture_boundaries(ax,layers.ArchitectureStage,xPlot,true);
    dx=max(1,0.02*(xPlot(end)-xPlot(1))); xlim(ax,[xPlot(1)-dx xPlot(end)+dx]); ylim(ax,[0 100]);
    [xTickPos,xTickLabels]=architecture_layer_ticks(layers.ArchitectureStage,xPlot);
    xticks(ax,xTickPos); xticklabels(ax,xTickLabels);
    yticks(ax,transform_prop_axis(yTickVals,yBreaksOrig,yBreaksVis)); yticklabels(ax,string(yTickVals));
    xlabel(ax,'Layer','FontName','Arial','FontSize',6,'Color','k');
    ylabel(ax,'Proportion of model units (%)','FontName','Arial','FontSize',6,'Color','k');
    style_axis(ax); hold(ax,'off');
    exportgraphics(fig,fullfile(figureRoot,slug+'_hierarchy.pdf'),'ContentType','vector');
    exportgraphics(fig,fullfile(figureRoot,slug+'_hierarchy.png'),'Resolution',600);
    close(fig);

    % --------------------- Unsmoothed hierarchy plot ----------------------
    fig=figure('Color','w','Position',[100 100 420 250],'Renderer','painters');
    ax=axes(fig,'Position',[0.16 0.18 0.80 0.66]); hold(ax,'on');
    xPlot=architecture_x_positions(layers.ArchitectureStage);
    plot(ax,xPlot,transform_prop_axis(pct(:,1),yBreaksOrig,yBreaksVis),'-','LineWidth',1.5,'Color',colors(1,:));
    plot(ax,xPlot,transform_prop_axis(pct(:,2),yBreaksOrig,yBreaksVis),'-','LineWidth',1.5,'Color',colors(2,:));
    plot(ax,xPlot,transform_prop_axis(pct(:,3),yBreaksOrig,yBreaksVis),'-','LineWidth',1.5,'Color',colors(3,:));
    add_architecture_boundaries(ax,layers.ArchitectureStage,xPlot,true);
    dx=max(1,0.02*(xPlot(end)-xPlot(1))); xlim(ax,[xPlot(1)-dx xPlot(end)+dx]); ylim(ax,[0 100]);
    [xTickPos,xTickLabels]=architecture_layer_ticks(layers.ArchitectureStage,xPlot);
    xticks(ax,xTickPos); xticklabels(ax,xTickLabels);
    yticks(ax,transform_prop_axis(yTickVals,yBreaksOrig,yBreaksVis)); yticklabels(ax,string(yTickVals));
    xlabel(ax,'Layer','FontName','Arial','FontSize',6,'Color','k');
    ylabel(ax,'Proportion of model units (%)','FontName','Arial','FontSize',6,'Color','k');
    style_axis(ax); hold(ax,'off');
    exportgraphics(fig,fullfile(figureRoot,slug+'_hierarchy_nosmooth.pdf'),'ContentType','vector');
    exportgraphics(fig,fullfile(figureRoot,slug+'_hierarchy_nosmooth.png'),'Resolution',600);
    close(fig);
end
n=height(models); ncol=4; nr=ceil(n/ncol);
fig=figure('Color','w','Position',[100 100 195*ncol 126*nr],'Renderer','painters');
tl=tiledlayout(fig,nr,ncol,'TileSpacing','compact','Padding','compact');
for im=1:n
    ax=nexttile(tl); f=modelFrames{im}; hold(ax,'on');
    ys=[f.Pct_Ponly f.Pct_Sonly f.Pct_Shared]; nL=height(f);
    ysSmooth=stagewise_smooth(ys,f.ArchitectureStage,smoothWidth);
    xPlot=architecture_x_positions(f.ArchitectureStage);
    plot(ax,xPlot,transform_prop_axis(ysSmooth(:,1),yBreaksOrig,yBreaksVis),'-','LineWidth',1.5,'Color',colors(1,:));
    plot(ax,xPlot,transform_prop_axis(ysSmooth(:,2),yBreaksOrig,yBreaksVis),'-','LineWidth',1.5,'Color',colors(2,:));
    plot(ax,xPlot,transform_prop_axis(ysSmooth(:,3),yBreaksOrig,yBreaksVis),'-','LineWidth',1.5,'Color',colors(3,:));
    add_architecture_boundaries(ax,f.ArchitectureStage,xPlot,false);
    dx=max(1,0.02*(xPlot(end)-xPlot(1))); xlim(ax,[xPlot(1)-dx xPlot(end)+dx]); ylim(ax,[0 100]);
    [xTickPos,xTickLabels]=architecture_layer_ticks(f.ArchitectureStage,xPlot);
    xticks(ax,xTickPos); xticklabels(ax,xTickLabels);
    yticks(ax,transform_prop_axis(yTickVals,yBreaksOrig,yBreaksVis)); yticklabels(ax,string(yTickVals));
    title(ax,models.Model(im),'FontName','Arial','FontSize',6,'FontWeight','normal','Color','k');
    style_axis(ax); hold(ax,'off');
end
xlabel(tl,'Layer','FontName','Arial','FontSize',6,'Color','k');
ylabel(tl,'Proportion of model units (%)','FontName','Arial','FontSize',6,'Color','k');
exportgraphics(fig,fullfile(figureRoot,sprintf('all_%d_models_hierarchy.pdf',n)),'ContentType','vector');
exportgraphics(fig,fullfile(figureRoot,sprintf('all_%d_models_hierarchy.png',n)),'Resolution',600);
close(fig);

% --------------------- All-model unsmoothed hierarchy --------------------
fig=figure('Color','w','Position',[100 100 195*ncol 126*nr],'Renderer','painters');
tl=tiledlayout(fig,nr,ncol,'TileSpacing','compact','Padding','compact');
for im=1:n
    ax=nexttile(tl); f=modelFrames{im}; hold(ax,'on');
    ys=[f.Pct_Ponly f.Pct_Sonly f.Pct_Shared];
    xPlot=architecture_x_positions(f.ArchitectureStage);
    plot(ax,xPlot,transform_prop_axis(ys(:,1),yBreaksOrig,yBreaksVis),'-','LineWidth',1.5,'Color',colors(1,:));
    plot(ax,xPlot,transform_prop_axis(ys(:,2),yBreaksOrig,yBreaksVis),'-','LineWidth',1.5,'Color',colors(2,:));
    plot(ax,xPlot,transform_prop_axis(ys(:,3),yBreaksOrig,yBreaksVis),'-','LineWidth',1.5,'Color',colors(3,:));
    add_architecture_boundaries(ax,f.ArchitectureStage,xPlot,false);
    dx=max(1,0.02*(xPlot(end)-xPlot(1))); xlim(ax,[xPlot(1)-dx xPlot(end)+dx]); ylim(ax,[0 100]);
    [xTickPos,xTickLabels]=architecture_layer_ticks(f.ArchitectureStage,xPlot);
    xticks(ax,xTickPos); xticklabels(ax,xTickLabels);
    yticks(ax,transform_prop_axis(yTickVals,yBreaksOrig,yBreaksVis)); yticklabels(ax,string(yTickVals));
    title(ax,models.Model(im),'FontName','Arial','FontSize',6,'FontWeight','normal','Color','k');
    style_axis(ax); hold(ax,'off');
end
xlabel(tl,'Layer','FontName','Arial','FontSize',6,'Color','k');
ylabel(tl,'Proportion of model units (%)','FontName','Arial','FontSize',6,'Color','k');
exportgraphics(fig,fullfile(figureRoot,sprintf('all_%d_models_hierarchy_nosmooth.pdf',n)),'ContentType','vector');
exportgraphics(fig,fullfile(figureRoot,sprintf('all_%d_models_hierarchy_nosmooth.png',n)),'Resolution',600);
close(fig);

alignment=vertcat(alignParts{:}); layerTable=vertcat(layerParts{:});
if exist(excelFile,'file'), delete(excelFile); end
writetable(alignment,excelFile,'Sheet','Alignment');
writetable(layerTable,excelFile,'Sheet','LayerProportions_TotalUnits');
writetable(human,excelFile,'Sheet','HumanReference');
fprintf('Saved figures: %s\nSaved alignment: %s\n',figureRoot,excelFile);
end
function models=discover_models(root)
d=dir(root); d=d([d.isdir]); d=d(~ismember({d.name},{'.','..'})); rows=cell(0,2);
for i=1:numel(d)
    if ~isempty(dir(fullfile(root,d(i).name,'model_unit_classification_*.mat')))
        s=readtable(fullfile(root,d(i).name,'classification_summary.csv'),'TextType','string');
        rows(end+1,:)={string(s.Model(1)),string(d(i).name)}; %#ok<AGROW>
    end
end
models=cell2table(rows,'VariableNames',{'Model','ModelSlug'});
end
function models=sort_models_by_key(models)
% Fixed alphabetical ModelSlug order; used for plotting and Excel output.
keyOrder=["baichuan_audio_base","baichuan_audio_instruct","covo_audio_chat", ...
    "firered_audio","freeze_omni","glm_4_voice","kimi_audio_7b_instruct", ...
    "minicpm_o_2_6","minicpm_o_4_5","qwen_audio","qwen_audio_chat", ...
    "qwen2_audio","qwen2_audio_instruct","qwen2_5_omni"];
[tf,rank]=ismember(models.ModelSlug,keyOrder);
if any(~tf)
    extra=models.ModelSlug(~tf);
    [~,extraOrd]=sort(lower(extra));
    extraRank=numel(keyOrder)+(1:numel(extraOrd));
    tmp=zeros(nnz(~tf),1);
    tmp(extraOrd)=extraRank;
    rank(~tf)=tmp;
end
[~,ord]=sort(rank);
models=models(ord,:);
end
function file=one_file(folder,pattern)
d=dir(fullfile(folder,pattern));
if numel(d)~=1, error('Expected one %s in %s.',pattern,folder); end
file=fullfile(d(1).folder,d(1).name);
end
function x=read_stat(file,name,nUnits)
x=squeeze(h5read(file,"/unit_statistics/"+name));
if iscell(x), sx=string(x); if all(ismember(upper(sx),["TRUE","FALSE"]),'all'), x=strcmpi(sx,"TRUE"); else, x=str2double(sx); end, end
if isvector(x), x=x(:); elseif size(x,1)~=nUnits&&size(x,2)==nUnits, x=x'; end
if size(x,1)~=nUnits, error('Unexpected %s shape.',name); end
end
function stage=infer_architecture_stage(layers)
n=height(layers); stage=strings(n,1);
vars=string(layers.Properties.VariableNames);
moduleVar="";
for candidate=["Module","ModuleName","ArchitectureStage","Stage"]
    hit=find(strcmpi(vars,candidate),1);
    if ~isempty(hit), moduleVar=vars(hit); break; end
end
for i=1:n
    name=lower(strtrim(string(layers.LayerName(i))));
    seq="";
    if ismember("SequenceType",vars), seq=lower(strtrim(string(layers.SequenceType(i)))); end
    module="";
    if strlength(moduleVar)>0, module=lower(strtrim(string(layers.(moduleVar)(i)))); end
    token=strjoin([module name seq]," ");
    if has_any(token,["decoder","language_model","language model","llm","qwen","llama","vicuna","glm","thinker","prefill"])
        stage(i)="Decoder";
    elseif has_any(token,["projector","projection","adapter","adaptor","connector","resampler","downsampl","pooler"])
        stage(i)="Adapter";
    elseif has_any(token,["conv","cnn","feature_extractor","feature extractor"])
        stage(i)="CNN";
    elseif has_any(token,["tokenizer","speech_token","audio_tokenizer","codec"])
        stage(i)="Audio tokenizer";
    elseif has_any(token,["q-former","qformer","q_former"])
        stage(i)="Q-Former";
    elseif has_any(token,["audio_encoder","audio encoder","speech_encoder","speech encoder","audio_block","audio block","encoder","audio_tower","audio tower","whisper","beats","audio_native"])
        stage(i)="Audio encoder";
    else
        stage(i)="Other";
    end
end
for i=2:n-1
    if stage(i)=="Other" && stage(i-1)==stage(i+1) && stage(i-1)~="Other", stage(i)=stage(i-1); end
end
if all(stage=="Other")
    warning('Could not infer architecture stages from layer metadata; plotting all layers as one stage.');
    stage(:)="Model layers";
end
end
function tf=has_any(textValue,patterns)
tf=any(contains(textValue,patterns,'IgnoreCase',true));
end
function sm=stagewise_smooth(values,stage,width)
values=double(values); sm=nan(size(values)); [starts,ends,~]=stage_runs(stage);
for ir=1:numel(starts)
    rows=starts(ir):ends(ir); w=min(width,numel(rows));
    if w<=1, sm(rows,:)=values(rows,:); else, sm(rows,:)=smoothdata(values(rows,:),1,'movmean',w); end
end
end
function add_architecture_boundaries(ax,stage,xPlot,showLabels)
[starts,ends,labels]=stage_runs(stage);
for ir=1:numel(ends)-1
    boundaryX=(xPlot(ends(ir))+xPlot(ends(ir)+1))/2;
    xline(ax,boundaryX,':','Color',[0.65 0.65 0.65],'LineWidth',0.5,'HandleVisibility','off');
end
if showLabels
    xMin=xPlot(1); xMax=xPlot(end); span=max(eps,xMax-xMin);
    for ir=1:numel(starts)
        center=(xPlot(starts(ir))+xPlot(ends(ir)))/2;
        xNorm=(center-xMin)/span;
        text(ax,xNorm,1.025,short_stage_label(labels(ir)),'Units','normalized', ...
            'HorizontalAlignment','center','VerticalAlignment','bottom','Clipping','off', ...
            'FontName','Arial','FontSize',5.5,'FontWeight','normal','Color',[0.25 0.25 0.25]);
    end
end
end
function [starts,ends,labels]=stage_runs(stage)
stage=string(stage(:)); n=numel(stage);
if n==0, starts=[]; ends=[]; labels=strings(0,1); return; end
change=[true; stage(2:end)~=stage(1:end-1)];
starts=find(change);
ends=[starts(2:end)-1; n];
labels=stage(starts);
end
function s=short_stage_label(s)
s=string(s);
switch s
    case "Audio encoder", s="Audio enc.";
    case "Audio tokenizer", s="Tokenizer";
    case "Adapter", s="Adapter";
    case "Decoder", s="Decoder";
    case "CNN", s="CNN";
    case "Q-Former", s="Q-Former";
    otherwise, s=char(s);
end
end
function print_architecture_summary(modelName,stage)
[starts,ends,labels]=stage_runs(stage);
parts=strings(numel(starts),1);
for i=1:numel(starts), parts(i)=sprintf('%s[%d-%d]',labels(i),starts(i),ends(i)); end
fprintf('    architecture: %s\n',strjoin(parts,' -> '));
end
function out=depth_bin(v,nBins)
v=double(v(:)); n=numel(v); out=nan(nBins,1); q=floor(n/nBins); r=mod(n,nBins); first=1;
for b=1:nBins
    count=q+(b<=r);
    if count>0, out(b)=mean(v(first:first+count-1),'omitnan'); first=first+count; end
end
good=find(isfinite(out));
if numel(good)==1, out(:)=out(good); elseif numel(good)>=2, out=interp1(good,out(good),1:nBins,'linear','extrap')'; end
end
function r=safe_corr(x,y,kind)
r=slm_plot_safe_corr(x,y,kind);
end
function yVis=transform_prop_axis(y,yBreaksOrig,yBreaksVis)
y=double(y); y=max(yBreaksOrig(1),min(yBreaksOrig(end),y));
yVis=interp1(yBreaksOrig,yBreaksVis,y,'linear');
end
function xPlot=architecture_x_positions(stage)
% Preserve layer order but expand a real encoder-to-decoder bridge/projector.
stage=string(stage(:)); nL=numel(stage); xPlot=1:nL;
encoderEnd=find(stage=="Audio encoder",1,'last');
decoderStart=find(stage=="Decoder",1,'first');
if isempty(encoderEnd) || isempty(decoderStart) || decoderStart<=encoderEnd, return; end
bridgeCount=decoderStart-encoderEnd-1;
if bridgeCount<=0, return; end
% Make the middle bridge visibly wider without changing layer order.
% A one-layer projector is expanded from 2 native layer-steps to 5 visual steps.
targetSpan=max(5,bridgeCount+3);
left=xPlot(encoderEnd); right=xPlot(decoderStart); nativeSpan=right-left;
extra=targetSpan-nativeSpan;
if extra<=0, return; end
bridgeIdx=(encoderEnd+1):(decoderStart-1);
if ~isempty(bridgeIdx)
    bridgeFrac=(1:numel(bridgeIdx))/(numel(bridgeIdx)+1);
    xPlot(bridgeIdx)=xPlot(bridgeIdx)+extra*bridgeFrac;
end
xPlot(decoderStart:end)=xPlot(decoderStart:end)+extra;
end
function [tickPos,tickLabels]=architecture_layer_ticks(stage,xPlot)
% Tick labels: CNN end, encoder end, decoder start (only if separated by a
% bridge/projector), and decoder end. Direct encoder->decoder uses one
% shared interface label to avoid two adjacent overlapping numbers.
stage=string(stage(:)); tickPos=[]; tickLabels=strings(0,1);
cnnEnd=find(stage=="CNN",1,'last');
encoderEnd=find(stage=="Audio encoder",1,'last');
decoderStart=find(stage=="Decoder",1,'first');
decoderEnd=find(stage=="Decoder",1,'last');
if ~isempty(cnnEnd)
    tickPos(end+1)=xPlot(cnnEnd); tickLabels(end+1)=string(cnnEnd);
end
if ~isempty(encoderEnd)
    tickPos(end+1)=xPlot(encoderEnd); tickLabels(end+1)=string(encoderEnd);
end
if ~isempty(encoderEnd) && ~isempty(decoderStart) && decoderStart>encoderEnd+1
    tickPos(end+1)=xPlot(decoderStart); tickLabels(end+1)=string(decoderStart);
end
if ~isempty(decoderEnd)
    tickPos(end+1)=xPlot(decoderEnd); tickLabels(end+1)=string(decoderEnd);
end
if isempty(tickPos)
    tickPos=xPlot(end); tickLabels=string(numel(stage));
end
[tickPos,ia]=unique(tickPos,'stable'); tickLabels=tickLabels(ia);
end
function style_axis(ax)
set(ax,'FontName','Arial','FontSize',6,'LineWidth',0.5,'XColor','k','YColor','k','TickDir','out','Box','off');
end
