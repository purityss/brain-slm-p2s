# Speech representation analysis in human sEEG and speech language models

English | [中文](README_中文.md)

This repository provides analysis code and intermediate data for **17 human sEEG participants** and **14 speech language models (SLMs)**. Both branches use the same 52 Mandarin words and factor-controlled phonological/semantic distance contrasts. They analyze temporal windows in neural contacts and sequence segments in model units, then examine brain–model alignment, transcription performance, and targeted interventions.

MAT, NPZ, and XLSX intermediates are managed with Git LFS. Run `git lfs install` before `git clone https://github.com/purityss/brain-slm-p2s.git` to retrieve the full files.

## Analysis guides

| Branch | Guide | Analyses |
|---|---|---|
| Human sEEG | [code_for_SEEG_data/README.md](code_for_SEEG_data/README.md) | From participant RDMs to contact classification, sequence, hierarchy, and geometry |
| Fourteen SLMs | [code_for_SLM_data/README.md](code_for_SLM_data/README.md) | From model activations and RDMs to unit classification, alignment, ASR, lesions, and steering |

Model Python environment setup is described in [requirements/README.md](code_for_SLM_data/requirements/README.md). Human group analyses use MATLAB R2024a; signal preprocessing uses EEGLAB/BIOSIG. Each branch guide follows the analysis order and provides methods, commands, outputs, and main and supplementary figure references.

## Figures and code

| Manuscript result | Human analysis | Model analysis |
|---|---|---|
| Fig. 1: shared design | 52 words, contact RDMs, classification | 52 words, unit RDMs, classification |
| Fig. 2: classes, local sequence, hierarchy | Figs. 2a,d,g; analyses related to S1/S2 | Figs. 2c,e,f,h,i; S3/S4 |
| Fig. 3: alignment versus performance | Group sequence and hierarchy references | Fourteen-model alignment and ASR; S5/S7/S8 |
| Fig. 4: targeted interventions | — | Lesion Fig. 4b and S9/S10; steering Figs. 4c,d |
| Fig. 5: dynamic geometry | Figs. 5a,d,g; S12 | Figs. 5b,c,e,f,h,i; S12–S24 |

## Data use

The repository includes de-identified participant RDMs, derived contact and region tables, model classification arrays, compact geometry caches, and figure summary tables. Raw participant data cannot be shared because of privacy and ethics requirements. Use of the derived neural data remains subject to the applicable data-use conditions.
