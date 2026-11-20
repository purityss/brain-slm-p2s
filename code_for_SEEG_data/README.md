# sEEG analysis

English | [中文](README_中文.md) | [Repository overview](../README.md) | [Model analysis](../code_for_SLM_data/README.md)

This directory follows the human-data analysis in the manuscript: high-γ
preprocessing, identification of speech-responsive contacts, contact-level
RDMs, phonological/semantic classification, and population analyses. The
single-participant scripts in `code_for_sub01/` document the upstream steps;
the group analysis runs on the 17 participant RDMs in this directory.

## 0. Environment and inputs

MATLAB R2024a was used to run the group scripts. Start MATLAB with
`code_for_SEEG_data` as the current directory. The upstream scripts also
require EEGLAB and its BIOSIG importer; statistical and MDS functions use
the Statistics and Machine Learning Toolbox. The supplied 17 participant RDMs
and brain-region table support group classification and plots. The model
pipeline has its own [environment and workflow guide](../code_for_SLM_data/README.md).

## 1. Preprocess high-γ responses

**Method.** The recording pipeline filters sEEG to 60–150 Hz, obtains the
Hilbert amplitude envelope, and epochs responses from −200 to 1200 ms around
speech onset. Trials with interictal spikes or recording artifacts require
manual inspection and exclusion.
The experiment has 52 words, each repeated 12
times in four blocks.

**Code.** `code_for_sub01/01_preprocessing/datacook_newword.m` illustrates
trigger handling, filtering, Hilbert amplitudes, and condition-wise epoching
for one participant. It expects four EDF recordings and the stimulus log.
This example stops with the `ALLEEG` array in memory; the curated HGA array
used by later scripts is `code_for_sub01/EEGLAB_sets/sub01_NB_150.mat`.

## 2. Identify speech-responsive contacts

**Method.** For each contact, trial-wise high-γ activity is standardized to
the −200 to −50 ms baseline. A one-sided one-sample t-test across pooled
trials compares the 0–800 ms post-onset response with baseline. P values are
corrected across contacts and time points within participant by BH-FDR;
responsive contacts require `q < 0.05` for at least 100 consecutive ms.

**Code.** `code_for_sub01/02_plot_response_responsive/Step1_plot_save_responsive_contacts.m`
reads the curated `ALLEEG` array and writes
`02_plot_response_responsive/final_responsive_q005_100ms/sub01_final_responsive_q005_100ms.mat`,
including `responsive_chans_col`. Its response plots are generated locally.
To run it after preparing `sub01_NB_150.mat`:

```matlab
run(fullfile('code_for_sub01','02_plot_response_responsive', ...
    'Step1_plot_save_responsive_contacts.m'))
```

Across the 17 participants, the analysis identifies 894 responsive contacts
from 1,909 recording sites (Results; Fig. S1). The upstream processing example
uses `code_for_sub01/`.

## 3. Construct contact-level RDMs

**Method.** For each responsive contact, high-γ amplitudes are normalized
using the baseline pooled across the 52 conditions and 12 repetitions.
The repetitions are averaged within each word. The resulting temporal
samples form a feature vector in each of 16 overlapping windows (200 ms
duration, 50 ms step; centers 50–800 ms). Cosine distances across the 52
words yield one 52 × 52 RDM per contact and window (Methods, Eq. 1).

**Code.** `code_for_sub01/03_construct_RDMs/step0_build_cosine_RDM_two_window_schemes.m`
reads the HGA array and the responsive-contact MAT from Step 2. It creates
`sub01_nchan_HGA_cosine_RDM_200ms_50ms.mat` containing `nchan_rdm`.
After Step 2:

```matlab
run(fullfile('code_for_sub01','03_construct_RDMs', ...
    'step0_build_cosine_RDM_two_window_schemes.m'))
```

All 17 primary RDMs are in `code_for_subXX/03_construct_RDMs/` and are read
by the group classifier below.

## 4. Assign brain regions and classify contacts (Fig. 1c, Fig. 2a pie chart; classification for Fig. S2)

The contact-to-region lookup is
`code_for_all_sub_group/01_arrange_ROI_MNI/all_sub_resp_contacts_new.mat`.
It links each of the 894 responsive contacts to an ROI. Electrode
localization and anatomical labeling use the MRI/CT, SPM12, Brainstorm,
FreeSurfer, and USC Brain atlas workflow described in Methods. The group code
reads this table.

**Method.** For every contact and window, the phonological contrast is the
mean distance for phonologically different/semantically different pairs
(650) minus phonologically matched/semantically different pairs (26).
The semantic contrast uses the same 650 different/different pairs minus
phonologically different/semantically matched pairs (650). Significance
uses 10,000 restricted stimulus-label permutations, a positive effect,
one-sided `p < 0.05`, and at least two consecutive significant windows.
Each contact is P-only, S-only, Shared, or None. Strength is the observed
contrast standardized against its permutation null distribution (Methods,
Eqs. 4–7).

**Code.** The script below reads the 17 RDMs and ROI lookup, performs the
classification, and writes the class counts and intermediate inputs for
the subsequent analyses:

```matlab
run(fullfile('code_for_all_sub_group','02_classify_units', ...
    'main_classify_factorControlled_with_downstream_exports.m'))
```

The output root is
`code_for_all_sub_group/02_classify_units/main_factorControlled_2consecutive_with_downstream/`.
Its `factorControlled2consecutiveZ_group_counts.csv` has 894 responsive
contacts: **263 P-only, 100 S-only, 65 Shared, and 466 None**, matching the
manuscript. The downstream MAT files contain the 16-window P/S strengths,
the contact/ROI class table, and ten ordered parts of the geometry RDM table.
Run `step04_num_phonology_semantics_pie.m` to draw the four-class composition
pie chart in Fig. 2a:

```matlab
run(fullfile('code_for_all_sub_group','03_plot_sequential_hierarchical_geometry', ...
    'step04_num_phonology_semantics_pie.m'))
```

## 5. Analyze sequential P2S dynamics (Fig. 2d)

**Method.** The 16-window P and S representation strengths from Step 4
are averaged across contacts within participant and then equally across
participants for each class. The script plots the class trajectories with
participant-level SEM. It also computes descriptive one-sided Wilcoxon
tests against zero and paired P-versus-S tests with cluster-permutation
correction for figure annotations (5,000 permutations; threshold 0.01).
These annotations are conditional on the same data used to classify contacts.

**Code.** `step01_plot_sequential_contacts_shadedSEM.m` reads the P/S
strength MAT files and generates the heatmaps and trajectories in
`results_01_sequential_shadedSEM/`.

```matlab
run(fullfile('code_for_all_sub_group','03_plot_sequential_hierarchical_geometry', ...
    'step01_plot_sequential_contacts_shadedSEM.m'))
```

## 6. Analyze the cortical hierarchy (Fig. 2g)

**Method.** For each of the 12 retained ROIs, the P-only, S-only, and
Shared proportions are calculated relative to **all responsive contacts
in that ROI**. The ROI order follows the median of participant-level median
distances to the ipsilateral pmHG. The supplied
`results_02_ROI_hierarchical_rankings/ROI_hierarchy_pmHG_distance_MEDIAN.csv`
fixes this order.
Spearman trends use raw proportions; the two-ROI moving average is for
display only.

**Code.** `step02_plot_hierarchical_12ROIs_3x3_rawAndSmooth.m` reads the
Step 4 contact/ROI table and the fixed rank CSV. Of the 894 responsive
contacts, 713 belong to these 12 ROIs. Outputs are generated in
`results_02_hierarchical/`.

```matlab
run(fullfile('code_for_all_sub_group','03_plot_sequential_hierarchical_geometry', ...
    'step02_plot_hierarchical_12ROIs_3x3_rawAndSmooth.m'))
```

## 7. Analyze representational geometry (Fig. 5a, d, g; Fig. S12a, d)

**Method.** For each class and window, contact RDMs are first averaged
within participant and then equally across participants. The group RDM is
normalized to [0, 1] for the displayed distance contrasts. Phonological
and semantic clustering use the same factor-controlled stimulus pairs as
Step 4. One-sided restricted-label permutation tests (10,000) are corrected
by Holm across all 16 windows separately for each dimension and class.
Two-dimensional nonmetric MDS of the group RDMs is aligned across windows by
unit-radius normalization, Procrustes alignment, and a common rotation
(Methods, “Sequence-resolved multidimensional scaling”).

**Code.** `step03_plotGeometryMDSorient_mache_model.m` joins the ten RDM
parts and displays the **100–300 ms** and **400–600 ms** windows; tests use
all 16 windows. The output directory is `results_03_geometry_match_model/`.

```matlab
run(fullfile('code_for_all_sub_group','03_plot_sequential_hierarchical_geometry', ...
    'step03_plotGeometryMDSorient_mache_model.m'))
```

