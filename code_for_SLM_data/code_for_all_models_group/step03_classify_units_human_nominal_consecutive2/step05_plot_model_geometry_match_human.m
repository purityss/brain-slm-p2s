function step05_plot_model_geometry_match_human(modelFilter,segmentNumbers,recomputeRDMs)
% MATLAB geometry analysis for the three classified model unit types.
%
% All displayed quantities are derived from one class-by-segment model RDM:
% units are averaged within layer and layers are then averaged equally.
% Bars show mean +/- SD across the relevant stimulus-pair distances.  The
% one-sided P/S p-values use the same restricted label-permutation nulls as
% the human classification analysis, followed by Holm correction across all
% four expanded segments (separately for P and S within each unit class).
%
% OUTPUT:
% Figures/cache:
% step03_sequential_hierarchy_geometry/matlab_results/03_geometry
% Excel:         step03_sequential_hierarchy_geometry/matlab_results/03_geometry/
%                matlab_geometry_statistics_meanRDM_permHolm4.xlsx
%
% ================================ USAGE ==================================
% 1) Run ALL available models:
%       step05_plot_model_geometry_match_human
%
% 2) Run ONE model only:
%       step05_plot_model_geometry_match_human("qwen2_audio")
%
%    modelFilter must exactly match the model folder name (ModelSlug), e.g.
%       "qwen2_audio"
%       "qwen_audio"
%       "minicpm_o_4_5"
%
% 3) Run several selected models:
%       step05_plot_model_geometry_match_human(["qwen2_audio","qwen_audio"])
%
% 4) Select any TWO expanded segment slots to display (default: [1 3]):
%       step05_plot_model_geometry_match_human("qwen2_audio",[2 4])
%
% 5) Force recomputation of the all-segment RDM cache:
%       step05_plot_model_geometry_match_human("qwen2_audio",[1 3],true)
%    This requires the full Step 02 source RDM as well as the classification
%    metadata; the compact caches are sufficient for the default false case.
%
%    Default:
%       recomputeRDMs = false
%    Existing MATLAB/Python all-segment RDM caches are reused when available.
%
% IMPORTANT:
% - If modelFilter is omitted, all available model folders are processed.
% - modelFilter uses ModelSlug (folder name), NOT the display model name.
% - Native segments are expanded to four slots as follows:
%       2 segments -> [1 1 2 2]
%       3 segments -> [1 2 3 3]
%       4 segments -> [1 2 3 4]

% Render figures fully in the background so batch runs never open windows.
previousFigureVisible=get(groot,'defaultFigureVisible');
visibilityCleanup=onCleanup(@()set(groot,'defaultFigureVisible',previousFigureVisible)); %#ok<NASGU>
set(groot,'defaultFigureVisible','off');
if nargin<1,modelFilter=strings(0,1);else,modelFilter=string(modelFilter);end
if nargin<2||isempty(segmentNumbers),segmentNumbers=[1 3];end
% Backward compatibility with the former (modelFilter,recomputeRDMs) call.
if islogical(segmentNumbers)&&isscalar(segmentNumbers)
 recomputeRDMs=segmentNumbers;segmentNumbers=[1 3];
elseif nargin<3
 recomputeRDMs=false;
end
segmentNumbers=double(segmentNumbers(:)');
if numel(segmentNumbers)~=2||any(segmentNumbers~=fix(segmentNumbers))|| ...
   any(segmentNumbers<1|segmentNumbers>4)||segmentNumbers(1)==segmentNumbers(2)
 error('segmentNumbers must contain two distinct integers from 1 to 4, e.g. [1 3].');
end
thisFile=mfilename('fullpath'); classificationRoot=fileparts(thisFile);
groupRoot=fileparts(classificationRoot); projectRoot=fileparts(groupRoot);
resultRoot=fullfile(groupRoot,'step03_sequential_hierarchy_geometry');
outputRoot=fullfile(resultRoot,'matlab_results','03_geometry');
figureRoot=fullfile(outputRoot,'figures'); cacheRoot=fullfile(outputRoot,'segment_rdms_mat');
pythonCacheRoot=fullfile(resultRoot,'results_03_geometry','segment_rdms_mat');
excelFile=fullfile(outputRoot,'matlab_geometry_statistics_meanRDM_permHolm4.xlsx');
if ~exist(figureRoot,'dir'),mkdir(figureRoot);end
if ~exist(cacheRoot,'dir'),mkdir(cacheRoot);end
rdmChunk=32; baseSeed=20260920;
numPermutations=10000; permutationSeed=20260918;

set(groot,'defaultAxesFontName','Arial','defaultTextFontName','Arial', ...
 'defaultAxesFontSize',7,'defaultTextFontSize',7,'defaultAxesLineWidth',0.5, ...
 'defaultAxesXColor','k','defaultAxesYColor','k');
classCodes=[1 2 3]; classNames=["P-only" "S-only" "Shared"];
classTags=["Ponly" "Sonly" "Shared"];
targetLabels="Segment "+string(segmentNumbers);
models=discover_models(classificationRoot); statParts=cell(0,1); statusParts=cell(0,1);
if ~isempty(modelFilter)
 missing=setdiff(modelFilter(:),string(models.ModelSlug),'stable');
 if ~isempty(missing),error('Missing classification MAT for requested model(s): %s',strjoin(missing,', '));end
 models=models(ismember(models.ModelSlug,modelFilter),:);
end
if isempty(models),error('No requested model classification directories were found.');end
[Pweights,Sweights]=restricted_permutation_weights(numPermutations,permutationSeed);

for im=1:height(models)
 slug=models.ModelSlug(im); modelName=models.Model(im); fprintf('[%d/%d] %s\n',im,height(models),modelName);
 modelDir=fullfile(classificationRoot,slug); classMat=one_file(modelDir,'model_unit_classification_*.mat');
 layers=readtable(fullfile(modelDir,'layer_metadata.csv'),'TextType','string','VariableNamingRule','preserve');
 layers=sortrows(layers,'LayerOrderZeroBased'); nUnits=sum(layers.Units);
 classCode=double(read_stat(classMat,'class_code',nUnits));
 unitLayer=double(read_stat(classMat,'layer_order_zero_based',nUnits));
 unitWithin=double(read_stat(classMat,'unit_index_within_layer_zero_based',nUnits));
 cacheFile=fullfile(cacheRoot,slug+'_class_segment_mean_rdms.mat');
 pyCache=fullfile(pythonCacheRoot,slug+'_class_segment_mean_rdms.mat'); cacheSource="MATLAB cache";
 if ~recomputeRDMs && valid_segment_cache(cacheFile)
   C=load(cacheFile);
 elseif ~recomputeRDMs && valid_segment_cache(pyCache)
   C=load(pyCache); save(cacheFile,'-struct','C','-v7.3'); cacheSource="existing Python cache";
 else
   metadataFile=fullfile(modelDir,'analysis_metadata.json');
   if ~isfile(metadataFile)
     error('Cannot rebuild the geometry cache: missing %s. Restore a valid cache, or provide the metadata and full Step 02 RDM.',metadataFile);
   end
   metadata=jsondecode(fileread(metadataFile));
   source=resolve_source_rdm(metadata,projectRoot);
   C=build_cache(source,layers,classCode,unitLayer,unitWithin,rdmChunk);
   save(cacheFile,'-struct','C','-v7.3'); cacheSource="recomputed from source RDM";
 end
 [rows,lookup]=geometry_statistics(modelName,slug,C,segmentNumbers,classCodes,classNames,Pweights,Sweights);
 statParts{end+1,1}=rows; %#ok<AGROW>
 statusParts{end+1,1}=table(modelName,slug,cacheSource,string(cacheFile), ...
   'VariableNames',{'Model','ModelSlug','CacheSource','CacheFile'}); %#ok<AGROW>
 group=double(C.segment_group_rdms(:,segmentNumbers,:,:));
 counts=double(C.segment_unit_counts(:,segmentNumbers));
 plot_geometry_model(modelName,slug,group,counts,lookup,figureRoot,classCodes,classNames,classTags,targetLabels,segmentNumbers,baseSeed+100*im);
end

stats=vertcat(statParts{:}); status=vertcat(statusParts{:});
if exist(excelFile,'file'),delete(excelFile);end
writetable(stats,excelFile,'Sheet','GeometryStatistics'); writetable(status,excelFile,'Sheet','CacheStatus');
fprintf('Saved figures: %s\nSaved statistics: %s\n',figureRoot,excelFile);
end

function C=build_cache(source,layers,classCode,unitLayer,unitWithin,chunk)
segmentSum=zeros(3,4,52,52); segmentLayers=zeros(3,4); segmentUnits=zeros(3,4);
layerSegments=cell(3,4);
for il=1:height(layers)
 order=double(layers.LayerOrderZeroBased(il)); nwin=double(layers.WindowCount(il));
 if nwin==2,segmentMap=[1 1 2 2];elseif nwin==3,segmentMap=[1 2 3 3];elseif nwin==4,segmentMap=1:4;else,error('Unsupported segment count: %d',nwin);end
 dsPath=char(layers.SourceDatasetPath(il)); nLayerUnits=double(layers.Units(il));
 for ic=1:3
   gi=find(unitLayer==order & classCode==ic); if isempty(gi),continue;end
   selected=sort(unitWithin(gi)); lm=mean_selected_rdms(source,dsPath,selected,nLayerUnits,nwin,chunk);
   for seg=1:4
     r=complete_rdm(squeeze(lm(segmentMap(seg),:,:)));
     segmentSum(ic,seg,:,:)=segmentSum(ic,seg,:,:)+reshape(r,[1 1 52 52]);
     segmentLayers(ic,seg)=segmentLayers(ic,seg)+1;
     segmentUnits(ic,seg)=segmentUnits(ic,seg)+numel(selected);
     layerSegments{ic,seg}=cat(1,layerSegments{ic,seg},reshape(r,[1 52 52]));
   end
 end
end
C=struct; C.cache_version=int32(4); C.class_codes=int32([1 2 3]); C.class_names={'P-only','S-only','Shared'};
group=nan(3,4,52,52,'single'); maxL=max([cellfun(@(x)size(x,1),layerSegments(:));1]); factors=nan(3,maxL,4,4,'single');
for ic=1:3
 for seg=1:4
   if segmentLayers(ic,seg)==0,continue;end
   group(ic,seg,:,:)=single(segmentSum(ic,seg,:,:)/segmentLayers(ic,seg));
   for il=1:size(layerSegments{ic,seg},1)
     g=squeeze(group(ic,seg,:,:)); off=g(~eye(52)); lo=min(off); hi=max(off); x=squeeze(layerSegments{ic,seg}(il,:,:));
     if hi>lo,x=(x-lo)/(hi-lo);else,x=zeros(52);end
     [pSame,bothDiff,sSame]=factor_means(x); factors(ic,il,seg,:)=[pSame bothDiff sSame bothDiff];
   end
 end
end
C.segment_numbers=int32(1:4); C.segment_labels={'Segment 1','Segment 2','Segment 3','Segment 4'};
C.segment_group_rdms=group; C.segment_layer_factor_values=factors;
C.segment_layer_counts=segmentLayers; C.segment_unit_counts=segmentUnits;
C.segment_expansion_rule='2 -> [1 1 2 2]; 3 -> [1 2 3 3]; 4 -> [1 2 3 4]';
C.aggregation_method='units averaged within layer, then all layers averaged equally for each expanded segment';
end

function lm=mean_selected_rdms(file,path,selected,nUnits,nwin,chunk)
info=h5info(file,path); dims=double(info.Dataspace.Size); nd=numel(dims);
unitDim=find(dims==nUnits,1); if isempty(unitDim),error('Cannot locate unit dimension in %s.',path);end
winDim=find(dims==nwin & (1:nd)~=unitDim,1); stimDims=setdiff(1:nd,[unitDim winDim],'stable');
if numel(stimDims)~=2||any(dims(stimDims)~=52),error('Unexpected RDM layout in %s.',path);end
acc=zeros(nwin,52,52); selected=double(selected(:));
for k=1:chunk:numel(selected)
 ids=selected(k:min(k+chunk-1,end)); lo=ids(1); hi=ids(end); start=ones(1,nd); count=dims;
 start(unitDim)=lo+1; count(unitDim)=hi-lo+1; raw=h5read(file,path,start,count);
 raw=permute(raw,[unitDim winDim stimDims]); raw=reshape(raw,[hi-lo+1 nwin 52 52]); rel=ids-lo+1;
 acc=acc+squeeze(sum(double(raw(rel,:,:,:)),1,'omitnan'));
end
lm=acc/numel(selected); for w=1:nwin,lm(w,:,:)=complete_rdm(squeeze(lm(w,:,:)));end
end

function [pSame,bothDiff,sSame]=factor_means(rdm)
% Retained in the cache for backward compatibility; inference below uses
% the cross-layer mean RDM rather than these layer-wise values.
[rr,cc]=find(tril(true(52),-1)); labels=[zeros(26,1);ones(26,1)]; cross=labels(rr)~=labels(cc); truePair=false(numel(rr),1);
for i=1:26,truePair=truePair|(rr==i+26 & cc==i);end
v=rdm(sub2ind([52 52],rr,cc));pSame=mean(v(truePair),'omitnan');bothDiff=mean(v(cross&~truePair),'omitnan');sSame=mean(v(~cross),'omitnan');
end

function [T,lookup]=geometry_statistics(model,slug,C,segmentNumbers,classCodes,classNames,Pweights,Sweights)
% Statistics use the single cross-layer mean RDM, not layers as samples.
group=double(C.segment_group_rdms);
lc=double(C.segment_layer_counts); uc=double(C.segment_unit_counts);
allLabels="Segment "+string(1:4); parts=cell(0,1); lookup=cell(3,2);
for ic=1:3
  means=nan(4,4); sds=nan(4,4); effects=nan(4,2); pp=nan(4,1); sp=nan(4,1);
  for seg=1:4
    r=normalize_rdm(squeeze(group(ic,seg,:,:)));
    [means(seg,:),sds(seg,:)]=factor_distance_stats(r);
    y=single(r(tril(true(52),-1)));
    poolP=double(y'*Pweights); poolS=double(y'*Sweights);
    effects(seg,:)=[poolP(1) poolS(1)];
    pp(seg)=(1+nnz(poolP(2:end)>=poolP(1)))/size(Pweights,2);
    sp(seg)=(1+nnz(poolS(2:end)>=poolS(1)))/size(Sweights,2);
    if abs(effects(seg,1)-(means(seg,2)-means(seg,1)))>1e-6 || ...
       abs(effects(seg,2)-(means(seg,4)-means(seg,3)))>1e-6
      error('Permutation contrast and displayed distance contrast disagree.');
    end
  end
  pa=holm_adjust(pp); sa=holm_adjust(sp);
  for seg=1:4
    m=means(seg,:); sd=sds(seg,:);
    row=table(model,slug,classCodes(ic),classNames(ic),uc(ic,seg),lc(ic,seg),allLabels(seg),seg, ...
      m(1),sd(1),m(2),sd(2),m(3),sd(3),m(4),sd(4),effects(seg,1),effects(seg,2), ...
      pp(seg),pa(seg),sp(seg),sa(seg),string('SD across stimulus-pair distances'),int32(size(Pweights,2)-1), ...
      'VariableNames',{'Model','ModelSlug','ClassCode','UnitClass','NUnits','NLayers','Target','SegmentNumber', ...
      'P_similar_mean','P_similar_SD','P_different_mean','P_different_SD','S_similar_mean','S_similar_SD', ...
      'S_different_mean','S_different_SD','PhonologicalEffect','SemanticEffect','P_p_raw','P_p_holm4', ...
      'S_p_raw','S_p_holm4','ErrorBarDefinition','NPermutations'});
    parts{end+1,1}=row; %#ok<AGROW>
    hit=find(segmentNumbers==seg);
    for h=hit,lookup{ic,h}=row;end
  end
end
T=vertcat(parts{:});
end

function r=normalize_rdm(r)
r=complete_rdm(r); mask=tril(true(52),-1); off=r(mask); off=off(isfinite(off));
if isempty(off),r=nan(52);return;end
lo=min(off);hi=max(off);
if hi>lo,r=(r-lo)/(hi-lo);else,r=zeros(52);end
r(1:53:end)=0;
end

function [m,sd]=factor_distance_stats(rdm)
[rr,cc]=find(tril(true(52),-1)); labels=[zeros(26,1);ones(26,1)];
cross=labels(rr)~=labels(cc); truePair=false(numel(rr),1);
for i=1:26,truePair=truePair|(rr==i+26 & cc==i);end
v=rdm(sub2ind([52 52],rr,cc));
groups={v(truePair),v(cross&~truePair),v(~cross),v(cross&~truePair)};
m=nan(1,4);sd=nan(1,4);
for k=1:4,m(k)=mean(groups{k},'omitnan');sd(k)=std(groups{k},0,'omitnan');end
end

function [Pweights,Sweights]=restricted_permutation_weights(B,seed)
% Exact restricted nulls used by the human factor-controlled analysis.
rng(seed,'twister'); n=52; lower=tril(true(n),-1);[row,col]=find(lower);np=numel(row);
pairIndex=zeros(n);for k=1:np,pairIndex(row(k),col(k))=k;end
semantic=uint8([ones(26,1);2*ones(26,1)]);cross=semantic(row)~=semantic(col);
trueP=false(np,1);for i=1:26,trueP(pairIndex(i+26,i))=true;end
Pweights=zeros(np,B+1,'single');pairings=zeros(26,B+1,'uint8');pairings(:,1)=uint8(1:26);
for b=2:B+1,pairings(:,b)=uint8(randperm(26));end
for b=1:B+1
  w=zeros(np,1,'single');w(cross)=single(1/650);p=double(pairings(:,b));
  for i=1:26,w(pairIndex(26+p(i),i))=single(-1/26);end
  Pweights(:,b)=w;
end
Sweights=zeros(np,B+1,'single');flips=false(26,B+1);
for b=2:B+1,flips(:,b)=rand(26,1)<0.5;end
for b=1:B+1
  labels=zeros(n,1,'uint8');
  for i=1:26
    if ~flips(i,b),labels(i)=1;labels(i+26)=2;else,labels(i)=2;labels(i+26)=1;end
  end
  within=labels(row)==labels(col);betweenDiff=~within&~trueP;
  w=zeros(np,1,'single');w(betweenDiff)=single(1/650);w(within)=single(-1/650);Sweights(:,b)=w;
end
end

function plot_geometry_model(model,slug,group,counts,lookup,out,codes,names,tags,targetLabels,segmentNumbers,seed)
% Plotting settings matched to the human geometry figure.
% Heatmap, bars and MDS all use the same cross-layer mean RDM.

individualPosition=[100 60 400 600];
% Three exact copies of the individual-class canvas placed side by side.
% A small header is added above the three content panels; it
% does not shrink or move any panel content in physical pixel units.
combinedHeaderHeight=35;
combinedPosition=[100 40 3*individualPosition(3) ...
    individualPosition(4)+combinedHeaderHeight];

rdmPanelSize=[0.22 0.235];
rdmRowY=0.72;
barPanelSize=[0.22 0.15];
barRowY=0.50;
mdsPanelSize=[0.83 0.42];
mdsRowY=0.06;
leftColX=0.18;
rightColX=0.50;
mdsLeftX=0.10;

geometryAlpha=0.05;
normalMarkerSize=12;
specialMarkerSize=42;

% Same three example pairs highlighted in the human geometry plot:
% 煎饼/肩膀 = 8/34; 蜜桔/密码 = 14/40; 蘑菇/模特 = 16/42.
specialPairs=[8 34;14 40;16 42];
specialShapes={'d','s','o'};

mdsTimeSpacing=1.8;
mdsXCompression=0.50;
mdsYStretch=1.30;

% Always use the complete unit-class names in figure labels.
displayNames=["Phonological units" "Semantic units" "Shared units"];
segmentTag=sprintf('Segment%d_Segment%d_meanRDM_permHolm4',segmentNumbers(1),segmentNumbers(2));

%% ======================== Individual class figures ======================
for ic=1:3
    fig=figure('Color','w','Position',individualPosition, ...
        'Renderer','painters','Visible','off');

    % ----------------------------- RDM row -------------------------------
    ax1=axes(fig,'Position',[leftColX rdmRowY rdmPanelSize(1) rdmPanelSize(2)]);
    plot_rdm_triangle(ax1,squeeze(group(ic,1,:,:)));
    title(ax1,targetLabels(1), ...
        'FontName','Arial','FontSize',7,'FontWeight','normal');

    ax2=axes(fig,'Position',[rightColX rdmRowY rdmPanelSize(1) rdmPanelSize(2)]);
    plot_rdm_triangle(ax2,squeeze(group(ic,2,:,:)));
    title(ax2,targetLabels(2), ...
        'FontName','Arial','FontSize',7,'FontWeight','normal');

    % Keep the right RDM the same size as the left after adding colorbar.
    ax2Pos=ax2.Position;
    cb=colorbar(ax2);
    cb.FontName='Arial';
    cb.FontSize=7;
    ax2.Position=ax2Pos;
    cb.Label.String='Normalized cosine distance';
    cb.Label.FontName='Arial';
    cb.Label.FontSize=7;

    % -------------------------- Distance row -----------------------------
    ax3=axes(fig,'Position',[leftColX barRowY barPanelSize(1) barPanelSize(2)]);
    plot_bars(ax3,lookup{ic,1},geometryAlpha);

    ax4=axes(fig,'Position',[rightColX barRowY barPanelSize(1) barPanelSize(2)]);
    plot_bars(ax4,lookup{ic,2},geometryAlpha);

    % ---------------------------- MDS row -------------------------------
    ax5=axes(fig,'Position',[mdsLeftX mdsRowY mdsPanelSize(1) mdsPanelSize(2)]);
    plot_mds(ax5,squeeze(group(ic,1,:,:)),squeeze(group(ic,2,:,:)), ...
        seed+10*ic,specialPairs,specialShapes,normalMarkerSize, ...
        specialMarkerSize,mdsTimeSpacing,mdsXCompression,mdsYStretch);

    sgtitle(fig,{sprintf('%s: %s',char(model),char(displayNames(ic))), ...
        sprintf('%s n=%s; %s n=%s',char(targetLabels(1)),comma_int(counts(ic,1)), ...
        char(targetLabels(2)),comma_int(counts(ic,2)))}, ...
        'FontName','Arial','FontSize',7,'FontWeight','bold');

    exportgraphics(fig,fullfile(out,slug+'_Geometry_'+tags(ic)+'_'+segmentTag+'.pdf'), ...
        'ContentType','vector');
    exportgraphics(fig,fullfile(out,slug+'_Geometry_'+tags(ic)+'_'+segmentTag+'.png'), ...
        'Resolution',600);
    close(fig);
end

%% ========================= Combined 3-class figure ======================
fig=figure('Color','w','Position',combinedPosition, ...
    'Renderer','painters','Visible','off');
panelScale=1/3;
contentScale=individualPosition(4)/combinedPosition(4);

for ic=1:3
    panelLeft=(ic-1)*panelScale;
    mapPos=@(x,y,w,h)[panelLeft+x*panelScale y*contentScale ...
        w*panelScale h*contentScale];

    % ----------------------------- RDM row -------------------------------
    ax1=axes(fig,'Position',mapPos(leftColX,rdmRowY,rdmPanelSize(1),rdmPanelSize(2)));
    plot_rdm_triangle(ax1,squeeze(group(ic,1,:,:)));
    title(ax1,sprintf('%s\n%s',char(displayNames(ic)),char(targetLabels(1))), ...
        'FontName','Arial','FontSize',7,'FontWeight','normal');

    ax2=axes(fig,'Position',mapPos(rightColX,rdmRowY,rdmPanelSize(1),rdmPanelSize(2)));
    plot_rdm_triangle(ax2,squeeze(group(ic,2,:,:)));
    title(ax2,targetLabels(2), ...
        'FontName','Arial','FontSize',7,'FontWeight','normal');

    % Match the individual figure's colorbar and preserve the RDM axis size.
    ax2Pos=ax2.Position;
    cb=colorbar(ax2);
    cb.FontName='Arial';
    cb.FontSize=7;
    ax2.Position=ax2Pos;
    cb.Label.String='Normalized cosine distance';
    cb.Label.FontName='Arial';
    cb.Label.FontSize=7;

    % -------------------------- Distance row -----------------------------
    ax3=axes(fig,'Position',mapPos(leftColX,barRowY,barPanelSize(1),barPanelSize(2)));
    plot_bars(ax3,lookup{ic,1},geometryAlpha);

    ax4=axes(fig,'Position',mapPos(rightColX,barRowY,barPanelSize(1),barPanelSize(2)));
    plot_bars(ax4,lookup{ic,2},geometryAlpha);

    % ---------------------------- MDS row -------------------------------
    ax5=axes(fig,'Position',mapPos(mdsLeftX,mdsRowY,mdsPanelSize(1),mdsPanelSize(2)));
    plot_mds(ax5,squeeze(group(ic,1,:,:)),squeeze(group(ic,2,:,:)), ...
        seed+10*ic,specialPairs,specialShapes,normalMarkerSize, ...
        specialMarkerSize,mdsTimeSpacing,mdsXCompression,mdsYStretch);
end

annotation(fig,'textbox',[0 contentScale 1 1-contentScale], ...
    'String',model+": representational geometry across unit classes", ...
    'HorizontalAlignment','center','VerticalAlignment','middle', ...
    'EdgeColor','none','FontName','Arial','FontSize',7,'FontWeight','bold');

exportgraphics(fig,fullfile(out,slug+'_Geometry_threeClasses_'+segmentTag+'.pdf'), ...
    'ContentType','vector');
exportgraphics(fig,fullfile(out,slug+'_Geometry_threeClasses_'+segmentTag+'.png'), ...
    'Resolution',600);
close(fig);
end

function plot_rdm_triangle(ax,r)
% Human-geometry display:
% lower triangle only, each RDM independently normalized to [0,1].
r=complete_rdm(r);
offMask=tril(true(52),-1);
off=r(offMask);
off=off(isfinite(off));

if isempty(off)
    d=zeros(52);
else
    lo=min(off);
    hi=max(off);
    if hi>lo
        d=(r-lo)/(hi-lo);
    else
        d=zeros(52);
    end
end

D=nan(52);
D(offMask)=d(offMask);

imagesc(ax,1:52,1:52,D,'AlphaData',isfinite(D));

set(ax, ...
    'Color','w', ...
    'YDir','reverse', ...
    'FontName','Arial', ...
    'FontSize',7, ...
    'TickDir','out', ...
    'TickLength',[0 0], ...
    'LineWidth',0.5, ...
    'Box','on', ...
    'XTick',[], ...
    'YTick',[]);

axis(ax,'square');
xlim(ax,[0.5 52.5]);
ylim(ax,[0.5 52.5]);

colormap(ax,rdm_cmap(256));
caxis(ax,[0 1]);
end

function plot_bars(ax,row,alpha)
% Mean +/- SD across stimulus-pair distances from the model-level RDM.
m=[ ...
    row.P_similar_mean ...
    row.P_different_mean ...
    row.S_similar_mean ...
    row.S_different_mean];

se=[ ...
    row.P_similar_SD ...
    row.P_different_SD ...
    row.S_similar_SD ...
    row.S_different_SD];

x=[1.00 1.68 2.82 3.50];

cols=[ ...
    203 43 35; ...
    234 139 134; ...
    2 88 166; ...
    112 166 202]/255;

hold(ax,'on');

for i=1:4
    bar(ax,x(i),m(i),0.52, ...
        'FaceColor',cols(i,:), ...
        'EdgeColor','none');

    errorbar(ax,x(i),m(i),se(i),'k', ...
        'LineStyle','none', ...
        'LineWidth',0.55, ...
        'CapSize',0);
end

% Same fixed normalized-distance display range as the human figure.
ylim(ax,[0 1]);
xlim(ax,[0.55 3.95]);

set(ax, ...
    'FontName','Arial', ...
    'FontSize',7, ...
    'TickDir','out', ...
    'LineWidth',0.5, ...
    'Box','off', ...
    'XTick',[], ...
    'XTickLabel',[], ...
    'YTick',[0 1], ...
    'XColor','k', ...
    'YColor','k');

ylabel(ax,'Normalized distance', ...
    'FontName','Arial', ...
    'FontSize',7);

yr=1;
hP=max(m(1:2)+se(1:2)) + 0.055*yr;
hS=max(m(3:4)+se(3:4)) + 0.055*yr;

% Restricted label-permutation p-values, Holm-corrected over four segments.
draw_sig_bracket(ax,x(1),x(2),hP,row.P_p_holm4,alpha);
draw_sig_bracket(ax,x(3),x(4),hS,row.S_p_holm4,alpha);

hold(ax,'off');
end

function draw_sig_bracket(ax,x1,x2,y,p,alpha)
if ~isfinite(p)
    return;
end

yl=ylim(ax);
% tick=0.018*(yl(2)-yl(1));
% 
% plot(ax,[x1 x1 x2 x2], ...
%     [y-tick y y y-tick], ...
%     '-', ...
%     'Color','k', ...
%     'LineWidth',0.55, ...
%     'HandleVisibility','off');

plot(ax,[x1 x2],[y y], ...
    '-', ...
    'Color','k', ...
    'LineWidth',0.5, ...
    'HandleVisibility','off');

if p<alpha
    if p<0.001
        txt='***';
    elseif p<0.01
        txt='**';
    else
        txt='*';
    end
else
    txt='n.s.';
end

text(ax, ...
    (x1+x2)/2, ...
    y+0.006*(yl(2)-yl(1)), ...
    txt, ...
    'HorizontalAlignment','center', ...
    'VerticalAlignment','bottom', ...
    'FontName','Arial', ...
    'FontSize',7);
end

function plot_mds(ax,early,late,seed,specialPairs,specialShapes, ...
    normalMarkerSize,specialMarkerSize,timeSpacing,xCompression,yStretch)
% Keep the model's EXISTING MDS computation/alignment, while matching the
% human geometry figure's visual scaling, points, lines, ellipses and arrow.

rng(seed);
a=metric_mds(early);

rng(seed+1);
b=metric_mds(late);

% Existing model Procrustes-like orthogonal alignment retained.
[U,~,V]=svd(b'*a,'econ');
b=b*(U*V');

a=a-mean(a);
b=b-mean(b);

rad=max([ ...
    sqrt(sum(a.^2,2)); ...
    sqrt(sum(b.^2,2)); ...
    1e-12]);

a=a/rad;
b=b/rad;

% Existing model orientation logic retained.
d=mean(b(1:26,:),1)-mean(b(27:52,:),1);

if norm(d)>1e-12
    th=-pi/2-atan2(d(2),d(1));
    R=[cos(th) -sin(th);sin(th) cos(th)];
    a=a*R;
    b=b*R;
end

X=zeros(52,2);
Y=zeros(52,2);

X(:,1)=a(:,1)*xCompression;
Y(:,1)=a(:,2)*yStretch;

X(:,2)=b(:,1)*xCompression+timeSpacing;
Y(:,2)=b(:,2)*yStretch;

cols=[20 174 181;243 169 55]/255;

hold(ax,'on');

% Connections first.
for i=1:52
    c=cols(1+(i>26),:);
    lineColor=0.62*c+0.38*[1 1 1];

    plot(ax, ...
        X(i,:), ...
        Y(i,:), ...
        '-', ...
        'Color',lineColor, ...
        'LineWidth',1, ...
        'HandleVisibility','off');
end

% Points on top.
for it=1:2
    for i=1:52
        c=cols(1+(i>26),:);
        [isSpecial,pairIdx]=special_word_membership(i,specialPairs);

        if isSpecial
            scatter(ax, ...
                X(i,it), ...
                Y(i,it), ...
                specialMarkerSize, ...
                c, ...
                'filled', ...
                'Marker',specialShapes{pairIdx}, ...
                'MarkerEdgeColor',c, ...
                'MarkerFaceColor',c, ...
                'LineWidth',0.6, ...
                'HandleVisibility','off');
        else
            scatter(ax, ...
                X(i,it), ...
                Y(i,it), ...
                normalMarkerSize, ...
                c, ...
                'filled', ...
                'MarkerEdgeColor','none', ...
                'HandleVisibility','off');
        end
    end
end

% Same ellipse geometry as the human plot.
ellipseW=1.10;
ellipseH=2.20*yStretch;

for it=1:2
    centerX=(it-1)*timeSpacing;

    rectangle( ...
        'Parent',ax, ...
        'Position',[ ...
        centerX-ellipseW/2, ...
        -ellipseH/2, ...
        ellipseW, ...
        ellipseH], ...
        'Curvature',[1 1], ...
        'EdgeColor','k', ...
        'LineStyle','-', ...
        'LineWidth',0.55);
end

% Same arrow geometry as the human plot; wording remains model-appropriate.
arrowY=-1.35*yStretch;

plot(ax, ...
    [-0.55 timeSpacing+0.55], ...
    [arrowY arrowY], ...
    '-', ...
    'Color','k', ...
    'LineWidth',0.8);

plot(ax, ...
    timeSpacing+0.55, ...
    arrowY, ...
    '>', ...
    'Color','k', ...
    'MarkerFaceColor','k', ...
    'MarkerSize',4);

text(ax, ...
    timeSpacing/2, ...
    arrowY-0.30*yStretch, ...
    'Sequence segment', ...
    'HorizontalAlignment','center', ...
    'VerticalAlignment','top', ...
    'FontName','Arial', ...
    'FontSize',7);

axis(ax,'equal');
axis(ax,'off');

xExtent=max(abs([X(:,1);X(:,2)-timeSpacing]));
xPad=max(0.75,xExtent+0.15);

xlim(ax,[-xPad timeSpacing+xPad]);
ylim(ax,[-1.78 1.28]*yStretch);

hold(ax,'off');
end

function [tf,pairIdx]=special_word_membership(wordIdx,specialPairs)
tf=false;
pairIdx=NaN;

for ip=1:size(specialPairs,1)
    if any(specialPairs(ip,:)==wordIdx)
        tf=true;
        pairIdx=ip;
        return;
    end
end
end

function y=metric_mds(r)
D=max(complete_rdm(r),0);opts=statset('MaxIter',1000,'TolFun',1e-6,'Display','off');y=mdscale(D,2,'Criterion','metricstress','Start','random','Replicates',8,'Options',opts);y=y-mean(y);rad=max(sqrt(sum(y.^2,2)));if rad>0,y=y/rad;end
end

function r=complete_rdm(x)
x=double(x);xt=x';r=nan(52);for i=1:52,for j=1:52,v=[x(i,j) xt(i,j)];r(i,j)=mean(v(isfinite(v)));end,end;r(1:53:end)=0;
end

function q=holm_adjust(p)
q=nan(size(p));good=find(isfinite(p));[~,o]=sort(p(good));run=0;for k=1:numel(o),i=good(o(k));run=max(run,min(1,(numel(good)-k+1)*p(i)));q(i)=run;end
end
function tf=valid_segment_cache(file)
tf=false;if ~isfile(file),return;end
required=["cache_version" "segment_group_rdms" "segment_layer_factor_values" "segment_layer_counts" "segment_unit_counts"];
try
 info=whos('-file',file); names=string({info.name});
 if ~all(ismember(required,names)),return;end
 v=load(file,'cache_version');tf=double(v.cache_version)==4;
catch
 tf=false;
end
end
function path=resolve_source_rdm(meta,projectRoot)
% Metadata was produced on Linux, but this analysis is also run locally on
% Windows.  Try the recorded path first, then an explicit relative path,
% and finally reconstruct a local path from the code_for_SLM_data suffix.
recorded=string(meta.source_rdm_file);
candidates=recorded;
if isfield(meta,'source_rdm_relative_to_data_root')
 relative=string(meta.source_rdm_relative_to_data_root);
 if strlength(relative)>0,candidates(end+1)=fullfile(projectRoot,relative);end
end
normalized=replace(recorded,"\","/");
token=regexp(char(normalized),'/code_for_SLM_data/(.+)$','tokens','once');
if ~isempty(token)
 localRelative=replace(string(token{1}),"/",filesep);
 candidates(end+1)=fullfile(projectRoot,localRelative);
end
for i=1:numel(candidates)
 if isfile(candidates(i)),path=candidates(i);return;end
end
error(['Cannot resolve the full Step 02 source RDM. Provide that RDM to rebuild the cache, ' ...
 'or restore a valid segment cache and set recomputeRDMs=false. Checked:\n  %s'], ...
 strjoin(candidates,newline+"  "));
end
function models=discover_models(root)
d=dir(root);d=d([d.isdir]);d=d(~ismember({d.name},{'.','..'}));rows=cell(0,2);for i=1:numel(d),if ~isempty(dir(fullfile(root,d(i).name,'model_unit_classification_*.mat'))),s=readtable(fullfile(root,d(i).name,'classification_summary.csv'),'TextType','string');rows(end+1,:)={string(s.Model(1)),string(d(i).name)};end,end %#ok<AGROW>
models=sortrows(cell2table(rows,'VariableNames',{'Model','ModelSlug'}),'ModelSlug');
end
function file=one_file(folder,pattern),d=dir(fullfile(folder,pattern));if numel(d)~=1,error('Expected one %s.',pattern);end;file=fullfile(d(1).folder,d(1).name);end
function x=read_stat(file,name,nUnits),x=squeeze(h5read(file,"/unit_statistics/"+name));if iscell(x),sx=string(x);if all(ismember(upper(sx),["TRUE","FALSE"]),'all'),x=strcmpi(sx,"TRUE");else,x=str2double(sx);end,end;if isvector(x),x=x(:);elseif size(x,1)~=nUnits&&size(x,2)==nUnits,x=x';end;if size(x,1)~=nUnits,error('Unexpected %s shape.',name);end;end
function cmap=rdm_cmap(n),nodes=[50 28 90;40 100 120;45 175 120;170 215 60;252 245 65]/255;cmap=interp1(linspace(0,1,5),nodes,linspace(0,1,n));end
function style_axis(ax),set(ax,'FontName','Arial','FontSize',7,'LineWidth',.5,'XColor','k','YColor','k','TickDir','out','Box','off');end
function s=comma_int(n),s=regexprep(sprintf('%d',n),'(?<=\d)(?=(\d{3})+$)',',');end
