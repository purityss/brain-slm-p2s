# 人脑 sEEG 与语音语言模型的语音表征分析

[English](README.md) | 中文

本仓库提供论文中 **17 名 sEEG 被试**与 **14 个语音语言模型（SLM）**的分析代码及中间数据。两条流程使用相同的 52 个普通话词和控制因素的音韵/语义距离对比，分别分析神经接触点的时间窗与模型单元的序列段，并研究脑–模型对齐、转写表现和定向干预。

MAT、NPZ 和 XLSX 中间数据使用 Git LFS 管理。克隆前运行 `git lfs install`，然后使用 `git clone https://github.com/purityss/brain-slm-p2s.git` 获取完整文件。

## 分析入口

| 分析线 | 说明 | 分析内容 |
|---|---|---|
| 人脑 sEEG | [code_for_SEEG_data/README_中文.md](code_for_SEEG_data/README_中文.md) | 从单被试 RDM 到接触点分类、序列、层级及几何分析 |
| 14 模型 SLM | [code_for_SLM_data/README_中文.md](code_for_SLM_data/README_中文.md) | 从模型激活和 RDM 到单元分类、对齐、ASR、病灶及 steering |

模型 Python 环境的配置见 [requirements/README_中文.md](code_for_SLM_data/requirements/README_中文.md)；人脑组级代码使用 MATLAB R2024a，信号处理使用 EEGLAB/BIOSIG。各目录 README 按分析顺序给出方法、运行命令、输出及主图和补图对应关系。

## 论文图与代码

| 论文结果 | 人脑分析 | 模型分析 |
|---|---|---|
| 图 1：共同实验与分析框架 | 52 词、接触点 RDM、分类 | 52 词、单元 RDM、分类 |
| 图 2：类别、局部序列、层级 | 图 2a、d、g；补图 S1、S2 相关分析 | 图 2c、e、f、h、i；补图 S3、S4 |
| 图 3：对齐与模型表现 | 序列、层级组级参照表 | 14 模型对齐指标与 ASR；补图 S5、S7、S8 |
| 图 4：定向干预 | — | 病灶图 4b、补图 S9、S10；steering 图 4c、d |
| 图 5：动态表征几何 | 图 5a、d、g；补图 S12 | 图 5b、c、e、f、h、i；补图 S12–S24 |

## 数据使用

仓库提供去标识化的单被试 RDM、接触点与脑区派生表、模型分类数组、紧凑几何缓存及绘图汇总表。被试原始数据因隐私与伦理要求无法公开；使用衍生神经数据时应遵循相应的数据使用条件。
