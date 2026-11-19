clear
close all

here = fileparts(mfilename('fullpath'));
load(fullfile(here,'all_models_num_units.mat'))

save_fig_path = fullfile(here,'main_figure_bar');
save_fig_name = 'all_models_prop_4unit';


%% 1. 提取模型名称

models = all_models_num_units(:,1);


%% 2. 提取四类units
% 顺序:
% P-only, S-only, Shared, None

plot_data_raw = cell2mat(all_models_num_units(:,[5,4,3,6]));


%% 3. 计算比例

row_totals = sum(plot_data_raw,2);
percentages = plot_data_raw ./ row_totals * 100;


%% 4. 颜色设置

color_p = [180,37,35]/255;       % P-only 红
color_s = [34,91,142]/255;       % S-only 蓝
color_both = [103,38,141]/255;   % Shared 紫
color_ns = [0.7 0.7 0.7];        % None 灰

% 对应 P-only, S-only, Shared, None
colors = [
    color_both
    color_s
    color_p
    color_ns
];


%% 5. Y轴非线性拉伸
% 0-10%区域放大到视觉高度的25%
% 10-50%映射到25-55%
% 50-100%映射到55-100%

transform_y = @(x) ...
    (x <= 10) .* (x * 2.5) + ...
    (x > 10 & x <= 50) .* (25 + (x - 10) * (30 / 40)) + ...
    (x > 50) .* (55 + (x - 50) * (45 / 50));

cum_perc = cumsum(percentages,2);
trans_cum = zeros(size(cum_perc));

for i = 1:size(cum_perc,1)
    for j = 1:size(cum_perc,2)
        trans_cum(i,j) = transform_y(cum_perc(i,j));
    end
end

trans_plot_data = [trans_cum(:,1), diff(trans_cum,1,2)];


%% 6. 绘制stacked bar

figure('Color','w','Position',[100 100 420 165])

b = bar(trans_plot_data,...
    'stacked',...
    'EdgeColor','none',...
    'BarWidth',0.65);


for k = 1:4
    b(k).FaceColor = colors(k,:);
end


%% 7. 坐标设置

set(gca,...
    'XTick',1:length(models),...
    'XTickLabel',models,...
    'TickLabelInterpreter','none',...
    'FontSize',6);

xtickangle(45)


% Y轴刻度

set(gca,...
    'YTick',[transform_y(0),...
             transform_y(10),...
             transform_y(50),...
             transform_y(100)],...
    'YTickLabel',{'0','10','50','100'})


ylabel({'Proportion of',' model units (%)'},'FontSize',6);

% title('Proportion of Unit Types across Models','FontSize',6)


%% 8. 美化

set(gca,...
    'FontSize',6,...
    'Box','off',...
    'FontName','Arial',...
    'TickDir','out',...
    'LineWidth',0.5,...
    'XColor','k',...
    'YColor','k');

grid off

set(gca,'Layer','top')

ylim([0 100])


%% 9. 保存

if ~exist(save_fig_path,'dir')
    mkdir(save_fig_path)
end

exportgraphics(gcf,...
    fullfile(save_fig_path,[save_fig_name,'.pdf']),...
    'Resolution',600);