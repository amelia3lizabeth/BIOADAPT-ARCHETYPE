# BIOADAPT-ARCHETYPE

**GMM-guided mixture-of-experts routing on biologically structured pathway
features for anti-PD-1 response prediction from tumour transcriptomics.**

This repository contains the code and processed data for the paper:

> **Biological heterogeneity, not modelling choice, governs the cross-cohort
> generalisation of transcriptomic response prediction in anti-PD-1 melanoma.**
> Elizabeth Amelia, Chayanit Piyawajanusorn, Pedro J. Ballester.
> *Bioinformatics* (under review), 2026.
>
> Department of Bioengineering, Imperial College London.
> Corresponding authors: Elizabeth Amelia (amelia.elizabeth17@imperial.ac.uk),
> Pedro J. Ballester (pedro.ballester@imperial.ac.uk).

---

## Overview

BIOADAPT-ARCHETYPE tests whether the failure of transcriptomic biomarkers to
reproduce across anti-PD-1 cohorts is driven by biological heterogeneity rather
than modelling choice. Patients are grouped into immune-microenvironment
archetypes by a Gaussian mixture model on biologically defined pathway axes;
archetype-specialised expert classifiers are trained and blended by soft
membership. The pipeline is evaluated under leave-one-cohort-out (LOCO) and
mixed-cohort nested cross-validation, and benchmarked against JADBio.

All normalisation, feature selection, routing, calibration, and thresholds are
fitted on training patients only.

## Repository structure

```
code/
  config.py                          Central configuration (cohorts, axes, hyperparameters)
  data_loader.py                     Leak-free loading, within-cohort normalisation, axis scores
  archetype_router.py                RIN transform + K=2 GMM soft-membership router
  loco_pathway_v5.py                 LOCO validation, pathway representation  (main LOCO results)
  pathway_aggregation_pipeline_v2.py Mixed-cohort nested CV, pathway representation
  pipeline_v2.py                     Mixed-cohort nested CV, gene representation
  run_loco_v2.py                     LOCO validation, gene representation
  smoke_test.py                      Fast end-to-end sanity check
data/
  I01_liu_raw.csv                    Liu et al. 2019
  I09_gide_raw.csv                   Gide et al. 2019
  I14_riaz_raw.csv                   Riaz et al. 2017 (pre-treatment samples only)
  I15_hugo_raw.csv                   Hugo et al. 2016
requirements.txt
```

## Installation

Python 3.12 recommended.

```bash
git clone https://github.com/amelia3lizabeth/BIOADAPT-ARCHETYPE.git
cd BIOADAPT-ARCHETYPE
pip install -r requirements.txt
```

Dependencies: numpy, pandas, scipy, scikit-learn, xgboost, optuna,
imbalanced-learn.

## Data

Four publicly available anti-PD-1 melanoma RNA-seq cohorts, harmonised to HGNC
gene symbols and restricted to the gene set shared across all four cohorts after
zero-variance filtering (7,421 genes). Response is binarised by RECIST v1.1
(responder = complete/partial response; non-responder = stable/progressive).

| ID | Study | n | Responders | Accession |
|----|-------|---|-----------|-----------|
| I01 | Liu et al. 2019   | 119 | 63 (53%) | dbGaP phs000452 (controlled access) |
| I09 | Gide et al. 2019  | 41  | 19 (46%) | ENA PRJEB23709 |
| I14 | Riaz et al. 2017  | 56  | 11 (20%) | GEO GSE91061 (pre-treatment only) |
| I15 | Hugo et al. 2016  | 28  | 15 (54%) | GEO GSE78220 |

Each CSV has one row per patient, lowercase gene columns, a `response` column
(0/1), a `cohort` column, and a `_normalisation` column (RPKM/FPKM/zscored).
Access to the Liu cohort (dbGaP phs000452) requires dbGaP authorisation.

## Reproducing the results

Scripts import sibling modules and read paths from `config.py`, so run them from
inside the `code/` directory:

```bash
cd code

python smoke_test.py                      # fast sanity check first

python loco_pathway_v5.py                 # LOCO, pathway   -> output/loco_pathway_v5_metrics.csv
python run_loco_v2.py                      # LOCO, gene      -> output/loco_gene_v2_metrics.csv
python pathway_aggregation_pipeline_v2.py  # mixed-cohort, pathway -> output/pathway_agg_v2_metrics.csv
python pipeline_v2.py                      # mixed-cohort, gene    -> output/gene_pipeline_v2_metrics.csv
```

The mixed-cohort scripts optionally accept fold indices, e.g.
`python pipeline_v2.py 2 3 4` runs folds 2–4 only.

Each run writes a metrics CSV (and a results pickle) to the output directory set
in `config.py`. The CSV contains every combination of objective, threshold
strategy, and calibration mode. **The numbers reported in the paper use
calibration = `isotonic_off`, the MCC objective, and the MCC threshold**, so
filter the CSV to `calibration == 'isotonic_off'` to match the manuscript. AUC
is invariant to calibration; MCC, sensitivity, and specificity are not.

Key settings in `config.py`: `RANDOM_SEED = 42`, `ARCHETYPE_K = 2`,
`N_BOOTSTRAP = 50`, `N_OPTUNA_TRIALS = 30`, `PRIMARY_COHORTS`. The active model
set for the reported pipelines is LR, RF, SVM, EN.

## Citation

```bibtex
@article{amelia_bioadapt_archetype_2026,
  title   = {Biological heterogeneity, not modelling choice, governs the
             cross-cohort generalisation of transcriptomic response prediction
             in anti-PD-1 melanoma},
  author  = {Amelia, Elizabeth and Piyawajanusorn, Chayanit and Ballester, Pedro J.},
  journal = {Bioinformatics},
  year    = {2026},
  note    = {Under review}
}
```

Archived release: [Zenodo DOI to be added on acceptance].

## Licence

MIT
