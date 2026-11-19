%% merge_sequential_hierarchy_alignment.m
% Merge sequential_alignment.xlsx and hierarchical_alignment.xlsx into one summary Excel.
%
% Input files (same folder as this script):
%   sequential_alignment.xlsx
%   hierarchical_alignment.xlsx
%
% Output:
%   combined_alignment_summary.xlsx
%
% Column order:
%   Model
%   ModelSlug
%   Sequential P-only
%   Sequential S-only
%   Sequential Shared
%   Hierarchy P-only
%   Hierarchy S-only
%   Hierarchy Shared

clearvars; clc;

%% Paths
here = fileparts(mfilename('fullpath'));
if isempty(here), here = pwd; end
seqFile = fullfile(here,'sequential_alignment.xlsx');
hierFile = fullfile(here,'hierarchical_alignment.xlsx');
outFile = fullfile(here,'combined_alignment_summary.xlsx');
assert(isfile(seqFile),'Cannot find: %s',seqFile);
assert(isfile(hierFile),'Cannot find: %s',hierFile);

%% Read Alignment sheets
seq = readtable(seqFile,'Sheet','Alignment','TextType','string','VariableNamingRule','preserve');
hier = readtable(hierFile,'Sheet','Alignment','TextType','string','VariableNamingRule','preserve');

requiredSeq = ["Model","ModelSlug","UnitClass"];
requiredHier = ["Model","ModelSlug","UnitClass"];
assert(all(ismember(requiredSeq,string(seq.Properties.VariableNames))),'Sequential Alignment sheet is missing required columns.');
assert(all(ismember(requiredHier,string(hier.Properties.VariableNames))),'Hierarchy Alignment sheet is missing required columns.');

% Automatically find the Pearson metric columns in the current files.
seqMetric = find_metric_column(seq,"SequentialPearson");
hierMetric = find_metric_column(hier,"HierarchyPearson");
fprintf('Sequential metric: %s\n',seqMetric);
fprintf('Hierarchy metric:  %s\n',hierMetric);

%% Use sequential file to define model order
base = unique(seq(:,{'Model','ModelSlug'}),'rows','stable');
seq.Model = string(seq.Model); seq.ModelSlug = string(seq.ModelSlug); seq.UnitClass = string(seq.UnitClass);
hier.Model = string(hier.Model); hier.ModelSlug = string(hier.ModelSlug); hier.UnitClass = string(hier.UnitClass);
base.Model = string(base.Model); base.ModelSlug = string(base.ModelSlug);

% Check that the two files contain the same model set.
missingInHier = setdiff(base.ModelSlug,unique(hier.ModelSlug),'stable');
extraInHier = setdiff(unique(hier.ModelSlug),base.ModelSlug,'stable');
assert(isempty(missingInHier),'Models missing from hierarchical_alignment.xlsx: %s',strjoin(missingInHier,', '));
assert(isempty(extraInHier),'Extra models in hierarchical_alignment.xlsx: %s',strjoin(extraInHier,', '));

%% Collect P-only / S-only / Shared values
classes = ["P-only","S-only","Shared"];
seqVals = nan(height(base),3);
hierVals = nan(height(base),3);

for ic = 1:3
    seqVals(:,ic) = get_class_metric(seq,base.ModelSlug,classes(ic),seqMetric);
    hierVals(:,ic) = get_class_metric(hier,base.ModelSlug,classes(ic),hierMetric);
end

%% Write final Excel in requested order
headers = [ ...
    "Model", ...
    "ModelSlug", ...
    seqMetric+"_P-only", ...
    seqMetric+"_S-only", ...
    seqMetric+"_Shared", ...
    hierMetric+"_P-only", ...
    hierMetric+"_S-only", ...
    hierMetric+"_Shared"];

data = cell(height(base),8);
data(:,1) = cellstr(base.Model);
data(:,2) = cellstr(base.ModelSlug);
data(:,3:5) = num2cell(seqVals);
data(:,6:8) = num2cell(hierVals);

if isfile(outFile), delete(outFile); end
writecell([cellstr(headers); data],outFile,'Sheet','AlignmentSummary');

%% Optional Excel formatting on Windows
% Use direct A1-style ranges rather than ws.Cells(...), which is more robust
% across MATLAB/Excel ActiveX versions.
if ispc
    excel=[]; wb=[];
    try
        excel=actxserver('Excel.Application');
        excel.DisplayAlerts=false;
        wb=excel.Workbooks.Open(outFile);
        ws=wb.Worksheets.Item('AlignmentSummary');
        lastRow=height(base)+1;
        dataRange=ws.Range(sprintf('A1:H%d',lastRow));
        headerRange=ws.Range('A1:H1');
        headerRange.Font.Bold=true;
        headerRange.Font.Color=16777215;          % white
        headerRange.Interior.Color=7949855;       % dark blue, RGB(31,78,121)
        headerRange.HorizontalAlignment=-4108;    % xlCenter
        ws.Range(sprintf('C2:H%d',lastRow)).NumberFormat='0.0000';
        dataRange.Columns.AutoFit;
        % Add filter arrows without creating a ListObject/Table object.
        try
            invoke(dataRange,'AutoFilter');
        catch
        end
        wb.Save();
        wb.Close(false);
        excel.Quit();
        delete(excel);
    catch ME
        warning('Excel styling skipped: %s',ME.message);
        try, if ~isempty(wb), wb.Close(false); end, catch, end
        try, if ~isempty(excel), excel.Quit(); delete(excel); end, catch, end
    end
end

fprintf('\nSaved:\n%s\n',outFile);

%% Local functions
function metric = find_metric_column(T,prefix)
names = string(T.Properties.VariableNames);
idx = find(startsWith(names,prefix,'IgnoreCase',true));
if isempty(idx)
    error('Cannot find a column starting with "%s".',prefix);
elseif numel(idx)>1
    combinedIdx = idx(contains(names(idx),"Combined",'IgnoreCase',true));
    if numel(combinedIdx)==1
        idx = combinedIdx;
    else
        error('Multiple "%s" columns found: %s',prefix,strjoin(names(idx),', '));
    end
end
metric = names(idx);
end

function values = get_class_metric(T,modelSlugs,className,metricName)
values = nan(numel(modelSlugs),1);
metric = double(T.(metricName));
for i = 1:numel(modelSlugs)
    use = T.ModelSlug==modelSlugs(i) & T.UnitClass==className;
    assert(nnz(use)==1,'Expected exactly one row for ModelSlug="%s", UnitClass="%s"; found %d.', ...
        modelSlugs(i),className,nnz(use));
    values(i) = metric(use);
end
end
