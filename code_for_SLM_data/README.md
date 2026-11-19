# Speech language model analysis (14 models)

English | [中文](README_中文.md) | [Repository overview](../README.md) | [Human sEEG analysis](../code_for_SEEG_data/README.md)

This guide follows the Methods dependency order: environment, 52-word activations, unit RDMs, P/S classification, sequential and hierarchical alignment, geometry, two-dataset ASR, lesioning, and steering. Run Python commands from `code_for_SLM_data/`; start MATLAB R2024a in this directory. The model and human analyses share the 52 words and factor-controlled contrasts, but use different units and sequence coordinates.

## 0. Environment and inputs

Set up the model-specific Conda environments and `ffmpeg` as described in [requirements/README.md](requirements/README.md), then configure model checkpoints, benchmark data, and runtime paths:

```bash
cd /path/to/brain-slm-p2s/code_for_SLM_data
export SLM_CONDA_ROOT=/path/to/conda/envs
export SLM_MODEL_ROOT=/path/to/model/checkpoints
export SLM_BENCHMARK_ROOT=/path/to/benchmarks
```

The repository includes classification NPZ/MAT files, layer and class summaries, geometry caches, alignment and ASR workbooks, lesion-summary workbooks, and steering summaries for two datasets.

## 1. Extract 52-word activations (Fig. 1b)

Use the same 52 Mandarin words as the human study and model-specific transcription prompts. PyTorch hooks capture convolutional channels, transformer hidden dimensions, intermediate modules where present, and decoder hidden dimensions at input-audio positions during prefill. Each unit retains its valid sequence. Model extractors are under `code_for_<Model>/01_extract_activation/`.

```bash
bash code_for_all_models_group/step01_run_extract_activations_14_models.sh --list
bash code_for_all_models_group/step01_run_extract_activations_14_models.sh \
  baichuan_audio_instruct freeze_omni
# Omit model keys to run all 14 after all paths and environments are configured.
```

Outputs are under each model's `01_extract_activation/<Model_Prefix>_hidden_dimension_activations/`, with a main `.npy` file of the same stem. This is the Step 02 input.

## 2. Build unit RDMs (Fig. 1c)

Split each layer's valid native sequence into adjacent, approximately equal, nonoverlapping segments: four for `T>7`, three for `T=6–7`, two for `T=4–5`. Cosine distances among the 52 words yield one `52×52` RDM per unit and segment; distances involving zero-norm vectors are undefined. The formal Mandarin input requires equal valid lengths within a layer across the 52 words.

```bash
python pipeline_code/step02_construct_rdms/validate_activation_inputs.py --models baichuan_audio_instruct
python code_for_all_models_group/step02_run_construct_adaptive_nonoverlap_unit_RDMs.py \
  --models baichuan_audio_instruct --dry-run --unit-chunk 128
python code_for_all_models_group/step02_run_construct_adaptive_nonoverlap_unit_RDMs.py \
  --models baichuan_audio_instruct --overwrite --unit-chunk 128
```

Use `--resume` instead of `--overwrite` after interruption if inputs have not changed; omit `--models` for all 14. Outputs are in each model's `02_construct_RDMs/adaptive_nonoverlap/`.

## 3. Classify P-only, S-only, and Shared units (Figs. 1c, 2c)

The phonological contrast compares 650 pairs differing in both factors with 26 phonologically matched, semantically different pairs. The semantic contrast compares those 650 pairs with 650 phonologically different, semantically matched pairs. For each native segment, 10,000 restricted label permutations produce one-sided p values and permutation-normalized P/S `Z` scores. A positive, significant effect in at least two adjacent native segments defines each dimension; the combination gives P-only, S-only, Shared, or None. The primary rule has no FDR correction.

```bash
python code_for_all_models_group/step03_classify_model_units_human_nominal_consecutive2.py \
  --data-root "$PWD" --models baichuan_audio_instruct \
  --permutations 10000 --device cuda --unit-chunk 64
```

Omit `--models` for all models. Outputs under `code_for_all_models_group/step03_classify_units_human_nominal_consecutive2/<model>/` include NPZ/MAT, `layer_metadata.csv`, `classification_summary.csv`, and `analysis_metadata.json`; `all_models_classification_summary.csv` combines all 14. In MATLAB, `step01_arrange_num_units.m` arranges counts and `step02_plot_bar_model_num_units.m` plots the **Fig. 2c** class proportions.

## 4. Sequential, hierarchical, and geometric representations (MDS; Figs. 2, 5)

MATLAB reads the supplied classification MAT files, group-level human references in `step00_human_reference/`, and compact geometry caches in `step03_sequential_hierarchy_geometry/matlab_results/03_geometry/segment_rdms_mat/`. First compute alignment for **all 14 models**, then plot representative geometry:

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

- **Sequence: Figs. 2e,f; S3–S4.** `step03_plot_model_sequential_shadedSEM.m` interpolates two, three, or four native segments to four relative positions. P/S scores are averaged across units within each layer, then equally across contributing layers. For alignment, the trajectory is interpolated to 16 human positions; P and S are separately Z-standardized and concatenated. Pearson `r` compares brain and model. Figures go to `matlab_results/01_sequential_shadedSEM/`; metrics to `matlab_results/sequential_alignment.xlsx`. A filtered run rewrites this workbook, so run all models before merging.
- **Hierarchy: Figs. 2h,i; S3–S4.** `step04_plot_model_hierarchy.m` divides each class count by all units in its layer, averages proportions within 12 consecutive depth bins, and computes Pearson `r` against 12 human regions ordered by ipsilateral pmHG distance. Smoothed curves are for display only. Figures go to `matlab_results/02_hierarchy/`; metrics to `matlab_results/hierarchical_alignment.xlsx`. The merge script writes `matlab_results/combined_alignment_summary.xlsx`.
- **Geometry: Figs. 5b,c,e,f,h,i; S12b,c,e,f and S13–S24.** `step05_plot_model_geometry_match_human.m` reads the compact 52-word RDM caches and displays segments 1 and 3. It averages units within layer, then layers equally; compares pair distances while controlling the other factor; and aligns two-dimensional MDS embeddings by unit radius and Procrustes. One-sided restricted permutation p values (10,000) receive Holm correction over four segments within each class and dimension. Figures go to `matlab_results/03_geometry/figures/`; statistics to `matlab_results/03_geometry/matlab_geometry_statistics_meanRDM_permHolm4.xlsx`.

`step03_build_model_geometry_segment_cache.py` generates geometry caches from classifications and Step 02 RDMs. The plotting script reads the existing caches in `matlab_results/03_geometry/segment_rdms_mat/`.

## 5. Evaluate ASR on two datasets (Table S3)

Use deterministic decoding on FLEURS-zh test (945 utterances) and WenetSpeech TEST_MEETING (8,370). Per-utterance accuracy is `max(0,1−CER)`, where CER counts character substitutions, deletions, and insertions divided by reference length. Average within each dataset, then average the two dataset accuracies equally for Fig. 3.

```bash
python code_for_all_models_group/step04_evaluate_asr_all_models.py \
  --dataset fleurs_zh --gpu 0 --retry-failures --continue-on-error
python code_for_all_models_group/step04_evaluate_asr_all_models.py \
  --dataset wenetspeech --subset test_meeting --gpu 0 --retry-failures --continue-on-error
python code_for_all_models_group/step04_summarize_fleurs_meeting_asr_metrics.py --overwrite
```

Per-utterance predictions and dataset summaries go under `step04_model_performance/fleurs_zh/` and `wenetspeech_test_meeting/`. The supplied final workbook is `step04_model_performance/fleurs_meeting_dataset_asr_metrics.xlsx`.

## 6. Relate alignment to performance (Fig. 3, Figs. S5/S7/S8)

The six Step 04 alignment indices are **brain–model Pearson `r` values**. Across models, the relationship between these indices and mean ASR accuracy is tested using **Spearman** correlation, BH-FDR over six tests, and leave-one-model-out checks. The subgroup script analyzes 11 audio-language and three omni models separately. Both inputs, `combined_alignment_summary.xlsx` and `fleurs_meeting_dataset_asr_metrics.xlsx`, are supplied in `step05_alignment_correlaion/`.

```matlab
run(fullfile(group,'step05_alignment_correlaion', ...
    'plot_alignment_performance_four_panels_with_Pearson_LOO_p.m'));
run(fullfile(group,'step05_alignment_correlaion', ...
    'plot_alignment_performance_by_model_group_with_Pearson_LOO_p.m'));
```

Outputs go to `step05_alignment_correlaion/matlab_results/`. The full-model script supports Fig. 3 and S5; the subgroup script supports S7/S8.

## 7. Targeted lesions (Fig. 4b, Figs. S9/S10)

Rank each class by its maximum common P/S magnitude across native segments and zero selected activations during inference. Encoder and adapter interventions use native acoustic positions; decoder interventions are restricted to input-audio positions during prefill. Matched doses across classes are 500, 1000, and so on, plus the exact endpoint `min(floor(0.01×total units), P-only count, S-only count, Shared count)`. Re-evaluate FLEURS-zh accuracy at each dose.

```bash
python code_for_all_models_group/step06_run_lesion_experiments.py --list
python code_for_all_models_group/step06_run_lesion_experiments.py \
  --models baichuan_audio_base qwen_audio --plan-only
python code_for_all_models_group/step06_run_lesion_experiments.py \
  --models baichuan_audio_base qwen_audio --gpu 0 --retry-failures
python code_for_all_models_group/step06_summary_lesion_results.py \
  --models baichuan_audio_base qwen_audio
```

The runner reads Step 03 NPZ, FLEURS-zh baselines, and each model's `06_lesion_experiment/model_spec.py`. Per-utterance outputs and `lesion_design.csv` go under `step06_lesion_results/model_details/`; the summarizer writes `lesion_summary/<model>.xlsx`. All **14 supplied workbooks** follow the dose rule. Replot without inference:

```matlab
run(fullfile(group,'step06_lesion_results', ...
    'step06_plot_lesion_results_from_summary_excel.m'));
```

The script plots all-model curves and maximum-dose bars (S9/S10), including the Baichuan-Audio-Base and Qwen-Audio examples for Fig. 4b.

## 8. Targeted activation steering (Figs. 4c,d)

For Baichuan-Audio-Base and Qwen2-Audio, reuse the Step 06 ranked `lesion_design.csv` prefixes without reranking. Select each model's matched endpoint count and scale selected activations as `h'=(1+0.05)h`. Evaluate **exactly correct transcripts/total utterances** on the 52 words and all 945 FLEURS-zh test utterances.

```bash
python code_for_all_models_group/step07_run_52words_steering.py \
  --models baichuan_audio_base qwen2_audio --plan-only
python code_for_all_models_group/step07_run_52words_steering.py \
  --models baichuan_audio_base qwen2_audio --gpu 0 --retry-failures
python code_for_all_models_group/step07_run_fleurs_zh_steering.py \
  --models baichuan_audio_base qwen2_audio --gpu 0 --retry-failures
```

Full runs write per-utterance and summary records. The repository includes `step07_steering_results/52words/summary/all_models_summary.csv` and `fleurs_zh/summary/all_models_summary.csv` endpoint tables. Plot Figs. 4c,d:

```matlab
run(fullfile(group,'step07_steering_results', ...
    'step07_plot_maxdose_exact_match.m'));
```

Figures go to `step07_steering_results/matlab_results/`.

