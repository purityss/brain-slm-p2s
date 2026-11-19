# Runtime profiles for the SLM code

English | [中文](README_中文.md)

These files record the Python dependencies used by the active 14-model
pipeline. Model families need separate environments because their Torch and
Transformers requirements differ. The main scripts refer to `py310_SLM`,
`py310_kimi`, `py310_covo`, and `py310_fireredaudio`; the Covo lesion runner
also accepts `py311_covo` and falls back to `py310_covo`.

| Profile | Use |
|---|---|
| `py310_SLM.txt` plus `conda_py310_SLM.txt` | Baichuan, Qwen, MiniCPM, Freeze-Omni, GLM, Qwen2.5-Omni |
| `py310_kimi.txt` | Kimi-Audio |
| `py310_covo.txt` or `py311_covo.txt` | Covo-Audio-Chat, according to upstream runtime |
| `py310_fireredaudio.txt` | FireRedAudio |
| `common.txt`, `covo_common.txt` | Shared dependencies imported by profiles |

From the `code_for_SLM_data` directory on Linux, create the four environments
expected by the Step 01 and Step 02 dispatchers:

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

If the Covo upstream runtime requires Python 3.11, also create its optional
profile for the Step 04/06 adapters:

```bash
conda create -n py311_covo python=3.11 pip -y
conda activate py311_covo
python -m pip install -r requirements/py311_covo.txt
conda deactivate
```

Install the Conda file before the corresponding pip profile. `ffmpeg` is an
executable dependency for audio preprocessing, not a pip substitute. Check
the selected PyTorch/CUDA build against the target GPU and driver, for example
with `python -c "import torch; print(torch.__version__, torch.version.cuda,
torch.cuda.is_available())"` inside each environment. The files
record dependencies only: model weights, upstream model repositories
(`model_runtime`), audio, and benchmark datasets must be obtained separately.
Some scripts still carry default paths from the original compute environment;
set their supported path options or environment variables when reproducing
upstream runs elsewhere.
