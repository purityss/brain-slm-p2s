clear all
close all

%% Save settings
save_fig_path = fullfile(fileparts(mfilename('fullpath')), 'results_04_pie');
save_fig_name = 'pie_3unit';
if ~exist(save_fig_path, 'dir'), mkdir(save_fig_path); end

%% Color settings
color_p = [180, 37, 35] / 255;       % Phonological - red
color_s = [34, 91, 142] / 255;       % Semantic - blue
color_both = [103, 38, 141] / 255;   % Both - purple
color_other = [0.7, 0.7, 0.7];       % Other - gray
colors = {color_p, color_s, color_both, color_other};

%% Text color settings
text_colors = {
    [1, 1, 1], ...   % Phonological - white
    [1, 1, 1], ...   % Semantic - white
    [1, 1, 1], ...   % Both - white
    [0, 0, 0]  ...   % Other - black
    };

%% Category counts from the classification output
group_counts_file = fullfile(fileparts(fileparts(mfilename('fullpath'))), ...
    '02_classify_units', 'main_factorControlled_2consecutive_with_downstream', ...
    'factorControlled2consecutiveZ_group_counts.csv');
assert(isfile(group_counts_file), 'Run the classification script first: %s', group_counts_file);
group_counts = readtable(group_counts_file);
row = strcmp(string(group_counts.window_scheme), '200ms_50ms');
assert(nnz(row) == 1, 'Expected one 200ms_50ms result row.');
count1 = group_counts.phonological_only(row);
count2 = group_counts.semantic_only(row);
count3 = group_counts.shared(row);
count4 = group_counts.none(row);
counts = [count1, count2, count3, count4];

%% Calculate percentages
total = sum(counts);
percentages = counts / total * 100;

%% Create percentage labels
labels = arrayfun(@(x) sprintf('%.1f%%', x), percentages, 'UniformOutput', false);

%% Figure settings
fig = figure('Units', 'inches', 'Position', [1, 1, 2.0, 2.0], ...
    'Color', 'w', 'Renderer', 'painters');
ax = axes('Parent', fig);

%% Plot pie chart with percentage labels
h = pie(ax, counts, labels);

%% Set slice colors and remove outlines; move labels inside slices
patch_count = 0;
text_count = 0;

for i = 1:length(h)
    if isa(h(i), 'matlab.graphics.primitive.Patch')
        patch_count = patch_count + 1;
        set(h(i), 'FaceColor', colors{patch_count}, ...
            'EdgeColor', 'none', 'LineStyle', 'none');

    elseif isa(h(i), 'matlab.graphics.primitive.Text')
        text_count = text_count + 1;

        pos = get(h(i), 'Position');
        pos(1:2) = pos(1:2) * 0.55;

        set(h(i), 'Position', pos, ...
            'FontName', 'Arial', ...
            'FontSize', 7, ...
            'FontWeight', 'normal', ...
            'Color', text_colors{text_count}, ...
            'HorizontalAlignment', 'center');
    end
end

%% Adjust axes
set(ax, 'FontName', 'Arial', 'FontSize', 7, ...
    'FontWeight', 'normal', 'XColor', 'k', 'YColor', 'k', ...
    'Position', [0.08, 0.08, 0.84, 0.84]);
axis(ax, 'equal');
axis(ax, 'off');

%% Print statistics
fprintf('Total: %d + %d + %d + %d = %d\n', counts, total);
fprintf('Percentages: %.1f%%, %.1f%%, %.1f%%, %.1f%%\n', percentages);

%% Export vector PDF
exportgraphics(fig, fullfile(save_fig_path, [save_fig_name, '.pdf']), ...
    'ContentType', 'vector', 'BackgroundColor', 'white');
