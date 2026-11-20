# sEEG 分析

[English](README.md) | 中文 | [仓库总览](../README_中文.md) | [模型分析](../code_for_SLM_data/README_中文.md)

本目录按照论文的人脑数据分析流程组织：高 γ 预处理、筛选语音响应接触点、
构建单接触点 RDM、分类音韵和语义接触点，以及群体表征分析。
`code_for_sub01/` 中的单被试代码展示上游处理；组级分析从目录内的
17 份单被试 RDM 开始运行。

## 0. 环境与输入

组级脚本已用 MATLAB R2024a 运行。执行下列命令时，将 MATLAB 当前目录
设为 `code_for_SEEG_data`。上游代码另需 EEGLAB 及其 BIOSIG 导入器；
统计和 MDS 函数使用 Statistics and Machine Learning Toolbox。所附 17 份单被试
RDM 和脑区表用于组级分类与绘图。模型侧从其独立的
[环境与流程](../code_for_SLM_data/README_中文.md)开始。

## 1. 高 γ 信号预处理

**方法。** 连续 sEEG 经 60–150 Hz 滤波和 Hilbert 变换得到高 γ 幅度包络，
再按声音起点截取 −200 至 1200 ms 的试次。含痫样放电或记录伪迹的试次
需人工检查并排除。实验包含 52 个词，
每词重复 12 次，共四个区块。

**代码。** `code_for_sub01/01_preprocessing/datacook_newword.m` 以一名被试
为例，展示触发标记处理、滤波、Hilbert 幅度及按条件分段。它读取四份
EDF 和刺激日志，执行后将 `ALLEEG` 留在工作区；后续脚本读取的整理后
HGA 文件为 `code_for_sub01/EEGLAB_sets/sub01_NB_150.mat`。

## 2. 筛选语音响应接触点

**方法。** 每个接触点的高 γ 反应以 −200 至 −50 ms 基线对每个试次进行
标准化。将 52 个词的试次合并，对 0–800 ms 的响应逐时间点进行单侧
单样本 t 检验。每名被试内对全部接触点和时间点作 BH-FDR 校正；
`q < 0.05` 且连续显著至少 100 ms 的接触点定义为语音响应接触点。

**代码。** `code_for_sub01/02_plot_response_responsive/Step1_plot_save_responsive_contacts.m`
读取整理后的 `ALLEEG`，将包含 `responsive_chans_col` 的结果保存到
`02_plot_response_responsive/final_responsive_q005_100ms/sub01_final_responsive_q005_100ms.mat`，
并在本地生成响应图。准备好 `sub01_NB_150.mat` 后运行：

```matlab
run(fullfile('code_for_sub01','02_plot_response_responsive', ...
    'Step1_plot_save_responsive_contacts.m'))
```

17 名被试共 1,909 个记录点，其中 894 个符合语音响应标准（Results；
Fig. S1）。本目录为这一上游步骤提供一名单被试的示例代码。

## 3. 构建单接触点 RDM

**方法。** 对每个响应接触点，使用全部 52 个条件、每条件 12 次重复的
基线样本估计标准化参数；先对同一词的重复试次求平均，再从 16 个重叠
时间窗提取时序特征。窗长 200 ms、步长 50 ms，中心为 50–800 ms。
对 52 个词两两计算余弦距离，每个接触点、每个时间窗得到一张 52 × 52
RDM（Methods，公式 1）。

**代码。** `code_for_sub01/03_construct_RDMs/step0_build_cosine_RDM_two_window_schemes.m`
读取 HGA 和第 2 步的响应接触点文件，生成只含 `nchan_rdm` 的
`sub01_nchan_HGA_cosine_RDM_200ms_50ms.mat`。完成第 2 步后运行：

```matlab
run(fullfile('code_for_sub01','03_construct_RDMs', ...
    'step0_build_cosine_RDM_two_window_schemes.m'))
```

17 份主要 RDM 位于各 `code_for_subXX/03_construct_RDMs/`，供下述组分类读取。

## 4. 脑区标注与接触点分类（图 1c、图 2a 的饼图；补图 S2 的分类基础）

`code_for_all_sub_group/01_arrange_ROI_MNI/all_sub_resp_contacts_new.mat`
记录了 894 个响应接触点的脑区。MRI/CT 配准、电极定位及解剖标签依据
Methods 所述 SPM12、Brainstorm、FreeSurfer 和 USC Brain atlas 流程获得。
组级代码从这份脑区表读取标签。

**方法。** 对每个接触点及时间窗，音韵效应为“音韵不同、语义不同”的
650 对词距减去“音韵相同、语义不同”的 26 对词距；语义效应为前一组
650 对减去“音韵不同、语义相同”的 650 对。采用 10,000 次受限刺激标签
置换、正向效应、单侧 `p < 0.05` 且至少两个相邻时间窗显著的判定标准，
将接触点分为 P-only、S-only、Shared 或 None。效应相对于置换零分布
标准化为表征强度 Z 值（Methods，公式 4–7）。

**代码。** 以下脚本读取 17 份 RDM 和脑区表，完成分类，并写出后续分析
所需的类别计数与中间文件：

```matlab
run(fullfile('code_for_all_sub_group','02_classify_units', ...
    'main_classify_factorControlled_with_downstream_exports.m'))
```

结果位于
`code_for_all_sub_group/02_classify_units/main_factorControlled_2consecutive_with_downstream/`。
其中 `factorControlled2consecutiveZ_group_counts.csv` 记录 894 个响应
接触点：**P-only 263、S-only 100、Shared 65、None 466**，与正文一致。
下游 MAT 包含 16 窗口的 P/S 强度、接触点类别及脑区表，以及按顺序分成
十份的几何 RDM 表。下列脚本绘制图 2a 中的四类比例饼图：

```matlab
run(fullfile('code_for_all_sub_group','03_plot_sequential_hierarchical_geometry', ...
    'step04_num_phonology_semantics_pie.m'))
```

## 5. 局部序列 P2S 分析（图 2d）

**方法。** 第 4 步得到的 16 窗口 P/S 表征强度在各被试内先对接触点
平均，再对被试等权平均，按类别绘制轨迹和被试层面的 SEM。脚本另对
零水平进行单侧 Wilcoxon 检验，对 P 与 S 进行配对检验，并用聚类置换
校正生成图中的显著性标注（5,000 次置换，阈值 0.01）。这些标注与分类
使用同一批数据，因此属于条件性描述统计。

**代码。** `step01_plot_sequential_contacts_shadedSEM.m` 读取 P/S 强度
MAT，在 `results_01_sequential_shadedSEM/` 中生成热图和轨迹：

```matlab
run(fullfile('code_for_all_sub_group','03_plot_sequential_hierarchical_geometry', ...
    'step01_plot_sequential_contacts_shadedSEM.m'))
```

## 6. 皮层层级分析（图 2g）

**方法。** 对保留的 12 个脑区，P-only、S-only 和 Shared 比例都以该
脑区的**全部语音响应接触点**为分母。脑区顺序依据各脑区到同侧 pmHG
距离的“被试内中位数的组中位数”。
`results_02_ROI_hierarchical_rankings/ROI_hierarchy_pmHG_distance_MEDIAN.csv`
固定了这一顺序。Spearman 趋势
使用原始比例，双脑区移动平均曲线只用于展示。

**代码。** `step02_plot_hierarchical_12ROIs_3x3_rawAndSmooth.m` 读取第 4 步
的接触点—脑区表及固定排序 CSV。894 个响应接触点中，有 713 个位于
这 12 个脑区。结果在 `results_02_hierarchical/` 中生成。

```matlab
run(fullfile('code_for_all_sub_group','03_plot_sequential_hierarchical_geometry', ...
    'step02_plot_hierarchical_12ROIs_3x3_rawAndSmooth.m'))
```

## 7. 表征几何分析（图 5a、d、g；补图 S12a、d）

**方法。** 对每类接触点和每个时间窗，先在被试内平均 RDM，再对被试
等权平均。展示距离对比时，将组 RDM 标准化到 [0, 1]。音韵与语义
聚类强度采用第 4 步相同的控制因素词对；10,000 次单侧受限标签置换
检验在每类、每个维度的 16 个时间窗内分别作 Holm 校正。二维非度量
MDS 经单位半径标准化、Procrustes 对齐和共同旋转，以比较不同时窗
的几何结构（Methods，“Sequence-resolved multidimensional scaling”）。

**代码。** `step03_plotGeometryMDSorient_mache_model.m` 按顺序合并十份 RDM，
显示 **100–300 ms** 和 **400–600 ms** 两个窗口；统计检验使用全部
16 个窗口。输出目录为 `results_03_geometry_match_model/`。

```matlab
run(fullfile('code_for_all_sub_group','03_plot_sequential_hierarchical_geometry', ...
    'step03_plotGeometryMDSorient_mache_model.m'))
```

