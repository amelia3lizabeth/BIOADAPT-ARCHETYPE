"""
BIOADAPT-ARCHETYPE v6 — Configuration
========================================
Changes vs previous run:
  - PRIMARY_COHORTS: I14_riaz added back. Class imbalance (20% responders)
    handled by SMOTE.
  - N_ARCHETYPE_OPTUNA_TRIALS: separate Optuna tuning for archetype experts
    on pathway features. Fixes hyperparameter mismatch flagged in review.
  - USE_STABLE_PRESCREEN: pathway prescreen computed as intersection of
    genes surviving variance filter in ALL folds (once, before the CV loop).
    Eliminates the fold-to-fold +/-1 gene variability.
"""
from pathlib import Path

ROOT       = Path(__file__).parent.parent
DATA_DIR   = ROOT / 'data'
OUTPUT_DIR = ROOT / 'output/spec_calibration_0point5'

COHORT_FILES = {
    'I01_liu':  DATA_DIR / 'I01_liu_raw.csv',
    'I09_gide': DATA_DIR / 'I09_gide_raw.csv',
    'I14_riaz': DATA_DIR / 'I14_riaz_raw.csv',
    'I15_hugo': DATA_DIR / 'I15_hugo_raw.csv',
}

# I14_riaz added back — SMOTE handles the 20% responder rate
PRIMARY_COHORTS = ['I01_liu', 'I09_gide', 'I14_riaz', 'I15_hugo']
LOCO_COHORTS    = ['I01_liu', 'I09_gide', 'I14_riaz', 'I15_hugo']

# ─── PATHWAY DEFINITIONS ─────────────────────────────────────────────────────
# Literature-derived only (not optimised on this dataset).
AXIS_DEFINITIONS = {
    'tumour':       ['epcam','krt8','krt18','krt19','cdh1'],
    'stromal':      ['fap','acta2','col1a1','col1a2','mfap2'],
    'ctl':          ['cd8a','cd8b','gzma','gzmb'],
    'terminal':     ['havcr2','entpd1','cxcl13','lag3','ctla4','eomes','klrg1'],
    'progenitor':   ['bach2','id3','il7r','cd28','ccr7','lef1'],
    'epigenetic':   ['ezh2','dnmt3a','hdac1','hdac2'],
    'mhc_i':        ['hla-a','hla-b','hla-c','b2m','nlrc5'],
    'b_cell':       ['cd19','ms4a1','cd79a','cd79b','cd22','pax5'],
    'antigen_pres': ['cd83','cd86','hla-drb1','hla-dqa1','hla-dpa1','ciita','flt3','itgax'],
    'ifn_gamma':    ['irf1','irf9','cxcl9','cxcl10','cxcl11','ido1','cd274'],
    'nk':           ['klrk1','ncr1','fcgr3a','klrb1','ncr3','nkg7','klrc1'],
    'myeloid':      ['cd68','cd163','csf1r','il10','mmp9','mmp2'],
    'treg':         ['foxp3','ikzf2','il2ra','nt5e'],
    'wnt_ctnnb1':   ['ctnnb1','axin1','apc','lef1','ccnd1','myc'],
    'proliferation':['mki67','ccnb1','ccna2','cdk1','pcna','mcm2','birc5'],
    'pi3k_akt':     ['pik3ca','pik3cb','akt1','akt2','mtor'],
    'hypoxia':      ['hif1a','ldha','pgk1','eno1','ca9'],
}

ARCHETYPE_AXES = ['mhc_i','b_cell','antigen_pres','ifn_gamma',
                   'terminal','ctl','nk','tumour','stromal','myeloid']

ARCHETYPE_K         = 2
ARCHETYPE_MIN_TRAIN = 15

# ─── CV ──────────────────────────────────────────────────────────────────────
N_OUTER_FOLDS  = 5
N_INNER_FOLDS  = 3

# ─── FEATURE SELECTION ───────────────────────────────────────────────────────
N_FEATURES_MI            = 150
VAR_FILTER_PCT           = 5
EXCLUDE_AXIS_GENES_FROM_POOL = True
PRESCREEN_MODE           = 'pathway'
N_FEATURES_PER_ARCHETYPE = 25
ARCHETYPE_BOOTSTRAP_THRESHOLD = 0.4

# Stable prescreen: intersection across ALL folds, computed once before loop
USE_STABLE_PRESCREEN = True

# ─── WITHIN-ARCHETYPE PCA ────────────────────────────────────────────────────
USE_ARCHETYPE_PCA   = False
N_PCA_PER_ARCHETYPE = 15

# ─── BOOTSTRAP ───────────────────────────────────────────────────────────────
N_BOOTSTRAP    = 50
BOOTSTRAP_FRAC = 0.8

# ─── OPTUNA ──────────────────────────────────────────────────────────────────
OPTUNA_OBJECTIVES         = ['MCC', 'F2']
N_OPTUNA_TRIALS           = 30
N_ARCHETYPE_OPTUNA_TRIALS = 20   # separate tuning for pathway-space experts
OPTUNA_TIMEOUT            = None

# ─── THRESHOLD STRATEGIES ────────────────────────────────────────────────────
THRESHOLD_STRATEGIES = ['MCC', 'F2', 'F3', 'SENS_CONSTRAINED']
F2_BETA    = 2.0
F3_BETA    = 3.0
SPEC_FLOOR = 0.50

# ─── CALIBRATION ─────────────────────────────────────────────────────────────
CALIBRATE_PROBA    = True
CALIBRATION_METHOD = 'isotonic'

# ─── MODELS ──────────────────────────────────────────────────────────────────
MODELS       = ['LR']
EXTRA_MODELS = ['RF', 'XGB']
USE_SMOTE    = True
SMOTE_K      = 5

# ─── VALIDATION ──────────────────────────────────────────────────────────────
RUN_GLOBAL_BASELINE = True
N_BOOTSTRAP_CI      = 1000
N_PERMUTATIONS      = 1000

# ─── REPRODUCIBILITY ─────────────────────────────────────────────────────────
RANDOM_SEED = 42
N_JOBS      = -1

# ─── LOGGING ─────────────────────────────────────────────────────────────────
VERBOSE          = True
LOGGING_ENABLED  = True
SAVE_CHECKPOINTS = True
SAVE_FOLD_MODELS = False
