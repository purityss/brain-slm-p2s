# SLM 代码的运行环境配置

[English](README.md) | 中文

这些文件记录当前 14 模型分析流程使用的 Python 依赖。不同模型系列要求不同版本的 Torch 和 Transformers，需使用独立环境。主要脚本引用 `py310_SLM`、`py310_kimi`、`py310_covo` 和 `py310_fireredaudio`；Covo 病灶运行脚本还接受 `py311_covo`，并可回退到 `py310_covo`。

| 配置文件 | 用途 |
|---|---|
| `py310_SLM.txt` 与 `conda_py310_SLM.txt` | Baichuan、Qwen、MiniCPM、Freeze-Omni、GLM、Qwen2.5-Omni |
| `py310_kimi.txt` | Kimi-Audio |
| `py310_covo.txt` 或 `py311_covo.txt` | Covo-Audio-Chat，依据上游运行库选择 |
| `py310_fireredaudio.txt` | FireRedAudio |
| `common.txt`、`covo_common.txt` | 被上述配置引用的共同依赖 |

在 Linux 的 `code_for_SLM_data` 目录执行下列命令，建立 Step 01 和 Step 02
调度脚本默认使用的四个环境：

```bash
conda create -n py310_SLM python=3.10 pip -y
conda install -n py310_SLM -c conda-forge \
  --file requirements/conda_py310_SLM.txt -y
conda activate py310_SLM
python -m pip install -r requirements/py310_SLM.txt
conda deactivate

conda create -n py310_kimi python=3.10 pip -y
conda activate py310_kimi
python -m pip install -r requirements/py310_kimi.txt
conda deactivate

conda create -n py310_covo python=3.10 pip -y
conda activate py310_covo
python -m pip install -r requirements/py310_covo.txt
conda deactivate

conda create -n py310_fireredaudio python=3.10 pip -y
conda activate py310_fireredaudio
python -m pip install -r requirements/py310_fireredaudio.txt
conda deactivate
```

如果 Covo 上游运行库要求 Python 3.11，可再为 Step 04/06 适配器建立
可选环境：

```bash
conda create -n py311_covo python=3.11 pip -y
conda activate py311_covo
python -m pip install -r requirements/py311_covo.txt
conda deactivate
```

先安装 Conda 文件，再安装对应的 pip 配置。音频预处理需要可执行程序
`ffmpeg`，不能用同名 pip 包替代。应在每个环境中运行
`python -c "import torch; print(torch.__version__, torch.version.cuda,
torch.cuda.is_available())"`，核对 PyTorch/CUDA 与目标 GPU、驱动的兼容性。
这些文件只记录依赖；模型权重、上游模型仓库（`model_runtime`）、
音频和基准数据需要另行获取。部分脚本仍保留原计算环境的默认路径，
在其他环境执行上游分析时应设置其支持的路径参数或环境变量。
