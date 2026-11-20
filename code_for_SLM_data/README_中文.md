# 语音语言模型分析（14 个模型）

[English](README.md) | 中文 | [仓库总览](../README_中文.md) | [人脑分析](../code_for_SEEG_data/README_中文.md)

本目录按论文 Methods 的依赖顺序说明环境、52 词激活、单元 RDM、P/S 分类、序列与层级对齐、几何、双数据集 ASR、病灶和 steering。下列 Python 命令从 `code_for_SLM_data/` 执行；MATLAB 命令在该目录启动 MATLAB R2024a 后执行。人脑侧使用同一组 52 词和分类对比，但分析单位是 sEEG 接触点及时间窗。

## 0. 环境与输入

按 [requirements/README_中文.md](requirements/README_中文.md) 配置模型专用 Conda 环境与 `ffmpeg`，并设置模型权重、基准语料和运行环境路径：

```bash
cd /path/to/brain-slm-p2s/code_for_SLM_data
export SLM_CONDA_ROOT=/path/to/conda/envs
export SLM_MODEL_ROOT=/path/to/model/checkpoints
export SLM_BENCHMARK_ROOT=/path/to/benchmarks
```

仓库提供 14 个模型的分类 NPZ/MAT、层与类别汇总、几何缓存、对齐与 ASR 工作簿、病灶汇总工作簿，以及两个数据集的 steering 汇总表。

## 1. 提取 52 词激活（图 1b）

使用与人脑实验相同的 52 个普通话词和模型专用转写提示。PyTorch hooks 记录音频编码器卷积通道、Transformer 隐维、可用的中间模块，以及解码器 prefill 输入音频 token 位置的隐维；保留每个单元的有效序列。各模型提取器在 `code_for_<Model>/01_extract_activation/`。

```bash
bash code_for_all_models_group/step01_run_extract_activations_14_models.sh --list
bash code_for_all_models_group/step01_run_extract_activations_14_models.sh \
  baichuan_audio_instruct freeze_omni
# 不指定模型键时运行全部 14 个模型。
```

输出在各模型 `01_extract_activation/<Model_Prefix>_hidden_dimension_activations/`，核心文件为同名 `.npy`；供第 2 步使用。

## 2. 构建单元 RDM（图 1c）

按层的原生有效序列长度分成相邻、近似等长的非重叠段：`T>7` 为 4 段，`T=6–7` 为 3 段，`T=4–5` 为 2 段。每个单元、每段计算 52 词间的余弦距离，得到 `52×52` RDM；零范数距离未定义。普通话正式输入要求同层 52 词的有效长度一致。

```bash
python pipeline_code/step02_construct_rdms/validate_activation_inputs.py --models baichuan_audio_instruct
python code_for_all_models_group/step02_run_construct_adaptive_nonoverlap_unit_RDMs.py \
  --models baichuan_audio_instruct --dry-run --unit-chunk 128
python code_for_all_models_group/step02_run_construct_adaptive_nonoverlap_unit_RDMs.py \
  --models baichuan_audio_instruct --overwrite --unit-chunk 128
```

中断且输入未变化时以 `--resume` 替代 `--overwrite`；不指定模型处理全部 14 个。输出在各模型 `02_construct_RDMs/adaptive_nonoverlap/`，含单元 RDM 与元数据。

## 3. 分类 P-only、S-only、Shared 单元（图 1c、图 2c）

音韵效应比较“音韵和语义都不同”的 650 对与“音韵相同、语义不同”的 26 对；语义效应比较前者与“音韵不同、语义相同”的 650 对。每个原生段运行 10,000 次受限标签置换，并将效应标准化为 P/S `Z`。正向效应、单侧显著且至少两个相邻原生段显著，决定四类标签；正式分类不做 FDR。

```bash
python code_for_all_models_group/step03_classify_model_units_human_nominal_consecutive2.py \
  --data-root "$PWD" --models baichuan_audio_instruct \
  --permutations 10000 --device cuda --unit-chunk 64
```

不指定 `--models` 运行全部模型。输出在 `code_for_all_models_group/step03_classify_units_human_nominal_consecutive2/<model>/`：NPZ/MAT、`layer_metadata.csv`、`classification_summary.csv`、`analysis_metadata.json`；根目录 `all_models_classification_summary.csv` 汇总 14 模型。MATLAB 的 `step01_arrange_num_units.m` 整理四类数量，`step02_plot_bar_model_num_units.m` 绘制**图 2c** 类别比例。

## 4. 序列、层级与表征几何（MDS；图 2、图 5）

MATLAB 从随仓库提供的分类 MAT、`step00_human_reference/` 人脑组级表和 `step03_sequential_hierarchy_geometry/matlab_results/03_geometry/segment_rdms_mat/` 缓存运行。先对**全部 14 模型**生成对齐表，再绘制两个代表模型：

```matlab
group = fullfile(pwd,'code_for_all_models_group');
classDir = fullfile(group,'step03_classify_units_human_nominal_consecutive2');
addpath(classDir);
run(fullfile(classDir,'step01_arrange_num_units.m'));
run(fullfile(classDir,'step02_plot_bar_model_num_units.m'));
step03_plot_model_sequential_shadedSEM;
step04_plot_model_hierarchy;
run(fullfile(group,'step03_sequential_hierarchy_geometry','matlab_results', ...
    'merge_sequential_hierarchy_alignment.m'));
step05_plot_model_geometry_match_human(["baichuan_audio_instruct","freeze_omni"],[1 3],false);
```

- **序列：图 2e、f；补图 S3、S4。** `step03_plot_model_sequential_shadedSEM.m` 将原生 2/3/4 段插值到 4 个相对位置。各类 P/S 强度先在层内平均单元，再对有该类单元的层等权平均；对齐时插值到人脑 16 窗口，分别 Z 标准化 P/S 并串接，以 Pearson `r` 比较脑与模型。图在 `matlab_results/01_sequential_shadedSEM/`，表为 `matlab_results/sequential_alignment.xlsx`。带模型键参数运行会重写该表；合并前必须无参数运行全部模型。
- **层级：图 2h、i；补图 S3、S4。** `step04_plot_model_hierarchy.m` 以每层全部模型单元为分母求三类比例，按深度分为约等长的 12 箱，并与以同侧 pmHG 距离排序的 12 脑区比例计算 Pearson `r`。平滑曲线只用于显示。图在 `matlab_results/02_hierarchy/`，表为 `matlab_results/hierarchical_alignment.xlsx`。合并脚本写出 `matlab_results/combined_alignment_summary.xlsx`。
- **几何：图 5b、c、e、f、h、i；补图 S12b、c、e、f 和 S13–S24。** `step05_plot_model_geometry_match_human.m` 读取 14 模型的紧凑缓存，默认展示段 1 与 3。先层内平均单元 RDM，再层间等权；计算控制另一因素的距离对比及二维度量 MDS，嵌入经单位半径和 Procrustes 对齐。10,000 次受限置换的单侧 p 值按类别、维度在 4 段内 Holm 校正。图在 `matlab_results/03_geometry/figures/`，统计表为 `matlab_results/03_geometry/matlab_geometry_statistics_meanRDM_permHolm4.xlsx`。

`step03_build_model_geometry_segment_cache.py` 用分类结果和 Step 02 RDM 生成几何缓存；绘图脚本读取 `matlab_results/03_geometry/segment_rdms_mat/` 中的现有缓存。

## 5. 双数据集 ASR（表 S3）

在 FLEURS-zh test（945 条）与 WenetSpeech TEST_MEETING（8,370 条）上确定性转写。每条准确率为 `max(0,1−CER)`，CER 为字符级替换、删除、插入错误数之和除以参考字符数；分别在数据集内平均，再对两套准确率等权平均以用于图 3。

```bash
python code_for_all_models_group/step04_evaluate_asr_all_models.py \
  --dataset fleurs_zh --gpu 0 --retry-failures --continue-on-error
python code_for_all_models_group/step04_evaluate_asr_all_models.py \
  --dataset wenetspeech --subset test_meeting --gpu 0 --retry-failures --continue-on-error
python code_for_all_models_group/step04_summarize_fleurs_meeting_asr_metrics.py --overwrite
```

逐条预测和数据集汇总写入 `step04_model_performance/fleurs_zh/` 与 `wenetspeech_test_meeting/`；最终 `step04_model_performance/fleurs_meeting_dataset_asr_metrics.xlsx` 已附。

## 6. 对齐与表现的关系（图 3，补图 S5、S7、S8）

第 4 步的六项对齐指标本身是**脑与模型之间的 Pearson `r`**；这里在模型之间比较对齐指标与两数据集平均 ASR 表现，使用 **Spearman** 相关、六项 BH-FDR，并做留一模型分析。第二个脚本分别分析 11 个 audio-language 和 3 个 omni 模型。`step05_alignment_correlaion/` 已包含所需的 `combined_alignment_summary.xlsx` 和 `fleurs_meeting_dataset_asr_metrics.xlsx`。

```matlab
run(fullfile(group,'step05_alignment_correlaion', ...
    'plot_alignment_performance_four_panels_with_Pearson_LOO_p.m'));
run(fullfile(group,'step05_alignment_correlaion', ...
    'plot_alignment_performance_by_model_group_with_Pearson_LOO_p.m'));
```

输出在 `step05_alignment_correlaion/matlab_results/`。主图 3 与补图 S5 由全模型脚本支持，补图 S7/S8 由亚组脚本支持。

## 7. 定向病灶（图 4b，补图 S9、S10）

各类别按跨原生段最大的共同 P/S 强度排序，推理时将目标单元激活置零。编码器/适配器作用于原生声学位置，解码器限于 prefill 音频位置。三类使用相同删除数；最大数为 `min(floor(总单元数×0.01),P-only 数,S-only 数,Shared 数)`，剂量为 500、1000、……加精确终点。每个剂量重新评估 FLEURS-zh 准确率。

```bash
python code_for_all_models_group/step06_run_lesion_experiments.py --list
python code_for_all_models_group/step06_run_lesion_experiments.py \
  --models baichuan_audio_base qwen_audio --plan-only
python code_for_all_models_group/step06_run_lesion_experiments.py \
  --models baichuan_audio_base qwen_audio --gpu 0 --retry-failures
python code_for_all_models_group/step06_summary_lesion_results.py \
  --models baichuan_audio_base qwen_audio
```

运行器读取第 3 步 NPZ、FLEURS-zh 基线、模型专用 `06_lesion_experiment/model_spec.py`，写出 `step06_lesion_results/model_details/` 逐条结果及 `lesion_design.csv`；汇总器生成 `lesion_summary/<model>.xlsx`。随仓库提供的 **14 份**工作簿已按上述剂量整理，可直接绘图：

```matlab
run(fullfile(group,'step06_lesion_results', ...
    'step06_plot_lesion_results_from_summary_excel.m'));
```

脚本生成全部模型病灶曲线与最大剂量柱图（补图 S9、S10）；图 4b 使用 Baichuan-Audio-Base 与 Qwen-Audio 的代表结果。

## 8. 定向激活 steering（图 4c、d）

对 Baichuan-Audio-Base 与 Qwen2-Audio 沿用 Step 06 `lesion_design.csv` 的三类排序，各取相同的最终上限数量，不重新排序。在指定位置按 `h'=(1+0.05)h` 放大激活；评价指标是 52 词及完整 945 条 FLEURS-zh 中的**完全正确转写数/总数**。

```bash
python code_for_all_models_group/step07_run_52words_steering.py \
  --models baichuan_audio_base qwen2_audio --plan-only
python code_for_all_models_group/step07_run_52words_steering.py \
  --models baichuan_audio_base qwen2_audio --gpu 0 --retry-failures
python code_for_all_models_group/step07_run_fleurs_zh_steering.py \
  --models baichuan_audio_base qwen2_audio --gpu 0 --retry-failures
```

完整运行会生成逐条记录和汇总。两张终点表位于 `step07_steering_results/52words/summary/all_models_summary.csv` 与 `fleurs_zh/summary/all_models_summary.csv`。绘制图 4c、d：

```matlab
run(fullfile(group,'step07_steering_results', ...
    'step07_plot_maxdose_exact_match.m'));
```

图在 `step07_steering_results/matlab_results/`。

