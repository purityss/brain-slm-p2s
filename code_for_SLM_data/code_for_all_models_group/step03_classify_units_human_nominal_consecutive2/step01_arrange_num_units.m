clear; clc;

%% load csv
here = fileparts(mfilename('fullpath'));
filename = fullfile(here,'all_models_classification_summary.csv');
T = readtable(filename);

%% model list
models = unique(T.Model,'stable');

num_models = length(models);

all_models_num_units = cell(num_models,6);

for i = 1:num_models
    
    model = models{i};
    idx = strcmp(T.Model,model);
    Ti = T(idx,:);
    
    % model name
    all_models_num_units{i,1} = model;
    
    % total units
    total_units = unique(Ti.TotalUnits);
    all_models_num_units{i,2} = total_units;
    
    pho = 0;
    sem = 0;
    shared = 0;
    
    for j = 1:height(Ti)
        
        class = Ti.Class{j};
        
        if strcmpi(class,'P-only')
            pho = Ti.Count(j);
            
        elseif strcmpi(class,'S-only')
            sem = Ti.Count(j);
            
        elseif strcmpi(class,'Shared')
            shared = Ti.Count(j);
        end
        
    end
    
    % save three classified groups
    all_models_num_units{i,3} = pho;
    all_models_num_units{i,4} = sem;
    all_models_num_units{i,5} = shared;
    
    % none = remaining units
    all_models_num_units{i,6} = total_units - pho - sem - shared;
    
end


%% convert numeric columns
all_models_num_units(:,2:end) = cellfun(@double,...
    all_models_num_units(:,2:end),...
    'UniformOutput',false);


%% save
save(fullfile(here,'all_models_num_units.mat'),'all_models_num_units');

disp('Finished: all_models_num_units.mat saved.');