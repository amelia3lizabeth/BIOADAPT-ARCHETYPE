"""
BIOADAPT-ARCHETYPE v6 — Pipeline
===========================================
Fixes applied following code review:

FIX-1  STABLE PRESCREEN           — unchanged
FIX-2  ARCHETYPE-SPECIFIC TUNING  — unchanged
FIX-3  COHORT-ARCHETYPE CONFOUND  — unchanged
FIX-4  BOOTSTRAP CI FIXED THRESH  — unchanged

STRUCTURAL ADDITIONS (for pipeline comparison with pathway version):
  PIPELINE_MODELS    : LR, RF, SVM, EN — in both baseline and routed
  CALIBRATION_MODES  : isotonic_on | isotonic_off | platt
  F2F3_MIN_SPEC      : FIX-1 spec floor in find_threshold
  TRAIN_HOLDOUT_FRAC : 20% of training set withheld before any training begins
  _gene_pipeline_eval: helper called twice per fold (dev->val, full_tr->te)
  get_oof            : raw OOF (calibration applied separately)
  fit_calibrator /
  apply_calibrator   : three-mode calibration

UNCHANGED (feature selection / dimensionality reduction):
  ArchetypeExpert (bootstrap MI selection, ps_idx, top_local)
  make_archetype_objective (pathway space LR tuning)
  compute_stable_pathway_prescreen
  build_pathway_prescreen
  bootstrap_ci / permutation_test
"""

import sys, time, logging, pickle, warnings
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression as _LogReg
from sklearn.model_selection import train_test_split as _tts
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (roc_auc_score, matthews_corrcoef,
                              balanced_accuracy_score, f1_score,
                              brier_score_loss, confusion_matrix)
from sklearn.pipeline import Pipeline
import xgboost as xgb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False

import config as cfg
from data_loader import (load_pooled, compute_axis_scores,
                          fit_train_normalisation, apply_train_normalisation,
                          fit_variance_filter)
from archetype_router import ArchetypeRouter


# ─── STRUCTURAL CONSTANTS ─────────────────────────────────────────────────────
# Mirror the pathway pipeline so results are directly comparable.
PIPELINE_MODELS    = ['LR', 'RF', 'SVM', 'EN']
CALIBRATION_MODES  = ['isotonic_on', 'isotonic_off', 'platt']
F2F3_MIN_SPEC      = 0.10   # FIX-1: prevents MCC=0 from near-all-positive thresholds
TRAIN_HOLDOUT_FRAC = 0.20   # fraction of fold training set withheld for unbiased train eval


# ─── LOGGING ─────────────────────────────────────────────────────────────────
def setup_logging(label=''):
    cfg.OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    fname = f'run{"_"+label if label else ""}.log'
    log   = logging.getLogger('bioadapt')
    log.setLevel(logging.DEBUG)
    log.handlers.clear()
    fh = logging.FileHandler(cfg.OUTPUT_DIR / fname, mode='w')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s', '%Y-%m-%d %H:%M:%S'))
    log.addHandler(fh)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter('%(message)s'))
    log.addHandler(ch)
    log.info(f"Log: {cfg.OUTPUT_DIR / fname}")
    return log

def get_logger(): return logging.getLogger('bioadapt')


# ─── METRICS & THRESHOLDS ────────────────────────────────────────────────────
TGRID = np.linspace(0.05, 0.95, 91)

def _fbeta(y, prob, t, beta):
    pred = (prob >= t).astype(int)
    tp = int(((pred==1)&(y==1)).sum()); fp = int(((pred==1)&(y==0)).sum())
    fn = int(((pred==0)&(y==1)).sum())
    prec = tp/max(tp+fp,1); rec = tp/max(tp+fn,1)
    return (1+beta**2)*prec*rec / max((beta**2*prec)+rec, 1e-9)

def find_threshold(y, prob, strategy):
    best_t, best_val = 0.5, -2.0
    for t in TGRID:
        pred = (prob >= t).astype(int)
        if pred.sum()==0 or pred.sum()==len(pred): continue
        tn=int(((pred==0)&(y==0)).sum()); fp=int(((pred==1)&(y==0)).sum())
        tp=int(((pred==1)&(y==1)).sum()); fn=int(((pred==0)&(y==1)).sum())
        spec=tn/max(tn+fp,1); sens=tp/max(tp+fn,1)
        # FIX-1: disallow near-all-positive thresholds for F2/F3
        if strategy in ('F2', 'F3') and spec < F2F3_MIN_SPEC:
            continue
        if   strategy=='MCC':              val=float(matthews_corrcoef(y,pred))
        elif strategy=='F2':               val=_fbeta(y,prob,t,2.0)
        elif strategy=='F3':               val=_fbeta(y,prob,t,3.0)
        elif strategy=='SENS_CONSTRAINED': val=sens if spec>=cfg.SPEC_FLOOR else -1.0
        else: raise ValueError(strategy)
        if val > best_val: best_val, best_t = val, float(t)
    return best_t, best_val

def compute_metrics(y, prob, threshold):
    pred = (prob >= threshold).astype(int)
    try:    tn,fp,fn,tp = confusion_matrix(y, pred, labels=[0,1]).ravel()
    except: tn=fp=fn=tp=0
    try:    auc = float(roc_auc_score(y, prob))
    except: auc = float('nan')
    sens=tp/max(tp+fn,1); spec=tn/max(tn+fp,1)
    return {
        'AUC':auc, 'MCC':float(matthews_corrcoef(y,pred)),
        'BACC':float(balanced_accuracy_score(y,pred)),
        'F1':float(f1_score(y,pred,zero_division=0)),
        'F2':float(_fbeta(y,prob,threshold,2.0)),
        'F3':float(_fbeta(y,prob,threshold,3.0)),
        'Sensitivity':float(sens), 'Specificity':float(spec),
        'Brier':float(brier_score_loss(y,prob)),
        'Threshold':float(threshold),
        'TP':int(tp),'TN':int(tn),'FP':int(fp),'FN':int(fn),
    }


# ─── CALIBRATION ─────────────────────────────────────────────────────────────
def fit_isotonic(oof_prob, oof_y):
    """Kept for bootstrap_ci / permutation_test backward compatibility."""
    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(oof_prob, oof_y)
    return iso

def fit_calibrator(oof_raw, y_tr, mode):
    """Fit one calibrator on raw OOF probs. Returns (calibrator_or_None, cal_oof)."""
    if mode == 'isotonic_off':
        return None, oof_raw.copy()
    elif mode == 'isotonic_on':
        iso = IsotonicRegression(out_of_bounds='clip')
        iso.fit(oof_raw, y_tr)
        return iso, iso.transform(oof_raw)
    elif mode == 'platt':
        cal = _LogReg(C=1.0, solver='lbfgs', max_iter=1000)
        cal.fit(oof_raw.reshape(-1, 1), y_tr)
        return cal, cal.predict_proba(oof_raw.reshape(-1, 1))[:, 1]
    else:
        raise ValueError(f'Unknown calibration mode: {mode}')

def apply_calibrator(calibrator, prob_raw, mode):
    """Apply a fitted calibrator to new probabilities."""
    if mode == 'isotonic_off' or calibrator is None:
        return prob_raw
    elif mode == 'isotonic_on':
        return calibrator.transform(prob_raw)
    elif mode == 'platt':
        return calibrator.predict_proba(prob_raw.reshape(-1, 1))[:, 1]
    else:
        raise ValueError(f'Unknown calibration mode: {mode}')


# ─── CLASSIFIERS ─────────────────────────────────────────────────────────────
def build_classifier(mname, params, y_tr):
    if mname == 'LR':
        return LogisticRegression(
            C=params['C'], max_iter=1000, class_weight='balanced',
            random_state=cfg.RANDOM_SEED)
    elif mname == 'EN':
        return LogisticRegression(
            C=params['C'], l1_ratio=params.get('l1_ratio', 0.5),
            penalty='elasticnet', solver='saga',
            max_iter=2000, class_weight='balanced',
            random_state=cfg.RANDOM_SEED)
    elif mname == 'RF':
        return RandomForestClassifier(
            n_estimators=params['n_estimators'], max_depth=params['max_depth'],
            max_features=params['max_features'], class_weight='balanced',
            random_state=cfg.RANDOM_SEED, n_jobs=cfg.N_JOBS)
    elif mname == 'SVM':
        return SVC(
            C=params['C'], kernel=params.get('kernel', 'rbf'),
            gamma='scale', probability=True, class_weight='balanced',
            random_state=cfg.RANDOM_SEED)
    else:  # XGB (kept for backward compat with cfg.MODELS)
        spw = float((y_tr==0).sum())/max((y_tr==1).sum(),1)
        return xgb.XGBClassifier(
            learning_rate=params['learning_rate'], n_estimators=params['n_estimators'],
            max_depth=params['max_depth'], subsample=params['subsample'],
            scale_pos_weight=spw, eval_metric='logloss',
            random_state=cfg.RANDOM_SEED, n_jobs=cfg.N_JOBS, verbosity=0)

def smote_data(X, y, smote_k, seed):
    if not (cfg.USE_SMOTE and HAS_SMOTE): return X, y
    k = min(smote_k, int(y.sum())-1, int((1-y).sum())-1)
    if k < 1: return X, y
    try:    return SMOTE(k_neighbors=k, random_state=seed).fit_resample(X, y)
    except: return X, y

def make_objective(mname, X_tr, y_tr, inner_cv, smote_k, objective):
    def obj_fn(trial):
        if mname == 'LR':
            clf = LogisticRegression(
                C=trial.suggest_float('C', 1e-3, 10, log=True),
                max_iter=1000, class_weight='balanced',
                random_state=cfg.RANDOM_SEED)
        elif mname == 'EN':
            clf = LogisticRegression(
                C=trial.suggest_float('C', 1e-3, 5.0, log=True),
                l1_ratio=trial.suggest_float('l1_ratio', 0.1, 0.9),
                penalty='elasticnet', solver='saga', max_iter=2000,
                class_weight='balanced', random_state=cfg.RANDOM_SEED)
        elif mname == 'RF':
            clf = RandomForestClassifier(
                n_estimators=trial.suggest_int('n_estimators', 50, 300),
                max_depth=trial.suggest_int('max_depth', 2, 8),
                max_features=trial.suggest_float('max_features', 0.1, 0.8),
                class_weight='balanced',
                random_state=cfg.RANDOM_SEED, n_jobs=cfg.N_JOBS)
        elif mname == 'SVM':
            clf = SVC(
                C=trial.suggest_float('C', 1e-2, 10.0, log=True),
                kernel=trial.suggest_categorical('kernel', ['linear', 'rbf']),
                gamma='scale', probability=True, class_weight='balanced',
                random_state=cfg.RANDOM_SEED)
        else:  # XGB
            spw = float((y_tr==0).sum())/max((y_tr==1).sum(),1)
            clf = xgb.XGBClassifier(
                learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                n_estimators=trial.suggest_int('n_estimators', 50, 300),
                max_depth=trial.suggest_int('max_depth', 2, 6),
                subsample=trial.suggest_float('subsample', 0.5, 1.0),
                scale_pos_weight=spw, eval_metric='logloss',
                random_state=cfg.RANDOM_SEED, n_jobs=cfg.N_JOBS, verbosity=0)
        vals = []
        for tri, vai in inner_cv.split(X_tr, y_tr):
            Xin, yin = smote_data(X_tr[tri], y_tr[tri], smote_k, cfg.RANDOM_SEED)
            pipe = Pipeline([('scl', StandardScaler()), ('clf', clf)])
            pipe.fit(Xin, yin)
            try:
                pv = pipe.predict_proba(X_tr[vai])[:, 1]
                _, v = find_threshold(y_tr[vai], pv, objective)
                vals.append(v)
            except Exception: vals.append(0.0)
        return float(np.mean(vals))
    return obj_fn

def get_oof(mname, X_tr, y_tr, params, inner_cv, smote_k):
    """Raw (uncalibrated) OOF probabilities. Calibration applied separately."""
    oof = np.zeros(len(y_tr))
    for tri, vai in inner_cv.split(X_tr, y_tr):
        Xin, yin = smote_data(X_tr[tri], y_tr[tri], smote_k, cfg.RANDOM_SEED)
        clf  = build_classifier(mname, params, yin)
        pipe = Pipeline([('scl', StandardScaler()), ('clf', clf)])
        pipe.fit(Xin, yin)
        oof[vai] = pipe.predict_proba(X_tr[vai])[:, 1]
    return oof

def get_oof_and_calibrate(mname, X_tr, y_tr, params, inner_cv, smote_k):
    """Kept for backward compatibility (LOCO baseline and bootstrap_ci callers)."""
    oof = get_oof(mname, X_tr, y_tr, params, inner_cv, smote_k)
    if cfg.CALIBRATE_PROBA:
        iso = fit_isotonic(oof, y_tr)
        return iso.transform(oof), iso
    return oof, None


# ─── FIX-2: ARCHETYPE-SPECIFIC HYPERPARAMETER TUNING ─────────────────────────
# UNCHANGED — tunes LR C in pathway-gene space only
def make_archetype_objective(X_pathway, y, inner_cv, smote_k):
    def obj_fn(trial):
        C = trial.suggest_float('C', 1e-4, 5.0, log=True)
        vals = []
        for tri, vai in inner_cv.split(X_pathway, y):
            Xin, yin = smote_data(X_pathway[tri], y[tri], smote_k, cfg.RANDOM_SEED)
            clf  = LogisticRegression(C=C, max_iter=1000, class_weight='balanced',
                                       random_state=cfg.RANDOM_SEED)
            pipe = Pipeline([('scl', StandardScaler()), ('clf', clf)])
            pipe.fit(Xin, yin)
            try:
                pv = pipe.predict_proba(X_pathway[vai])[:, 1]
                _, v = find_threshold(y[vai], pv, 'MCC')
                vals.append(v)
            except Exception: vals.append(0.0)
        return float(np.mean(vals))
    return obj_fn


# ─── ARCHETYPE EXPERT ─────────────────────────────────────────────────────────
# UNCHANGED — bootstrap MI selection, ps_idx, top_local all intact
class ArchetypeExpert:
    def __init__(self, mname, model_id=0):
        self.mname            = mname
        self.model_id         = model_id
        self.bootstrap_models = []
        self.freq_counter     = Counter()

    def fit(self, X, y, mem, gene_names, params, ps_idx=None, fold_id=0, arch_id=0):
        log  = get_logger()
        base = cfg.RANDOM_SEED + fold_id*10000 + arch_id*1000 + self.model_id*100
        rng  = np.random.RandomState(base)
        Xs   = X[:, ps_idx] if ps_idx is not None else X
        sg   = ([gene_names[i] for i in ps_idx]
                if ps_idx is not None else list(gene_names))
        n_select = min(cfg.N_FEATURES_PER_ARCHETYPE, Xs.shape[1])

        for b in range(cfg.N_BOOTSTRAP):
            n_samp = int(len(y)*cfg.BOOTSTRAP_FRAC)
            w_b    = mem / mem.sum()
            idx    = rng.choice(len(y), size=n_samp, replace=True, p=w_b)
            Xb, yb = Xs[idx], y[idx]
            if yb.sum() < 2 or (1-yb).sum() < 2: continue
            try:
                mib       = mutual_info_classif(Xb, yb, random_state=base+b)
                top_local = np.argsort(mib)[-n_select:]
                for ti in top_local: self.freq_counter[sg[ti]] += 1
                Xb_sel = Xb[:, top_local]
            except Exception as e:
                log.debug(f"  Bootstrap {b}: MI failed — {e}")
                top_local = np.arange(n_select)
                Xb_sel    = Xb[:, top_local]
            Xbs, ybs = smote_data(Xb_sel, yb, cfg.SMOTE_K, base+b)
            clf  = build_classifier(self.mname, params, ybs)
            pipe = Pipeline([('scl', StandardScaler()), ('clf', clf)])
            try:
                pipe.fit(Xbs, ybs)
                self.bootstrap_models.append((pipe, top_local))
            except Exception as e:
                log.debug(f"  Bootstrap {b}: fit failed — {e}")
        return self

    def predict_proba_raw(self, X, ps_idx=None):
        if not self.bootstrap_models: return np.full(X.shape[0], 0.5)
        Xs    = X[:, ps_idx] if ps_idx is not None else X
        probs = []
        for pipe, top_local in self.bootstrap_models:
            try:   probs.append(pipe.predict_proba(Xs[:, top_local])[:, 1])
            except: continue
        return np.mean(probs, axis=0) if probs else np.full(X.shape[0], 0.5)

    def stable_features(self, thr=None):
        if thr is None: thr = cfg.ARCHETYPE_BOOTSTRAP_THRESHOLD
        total = len(self.bootstrap_models)
        if total == 0: return []
        return [(g, cnt/total) for g, cnt in self.freq_counter.most_common()
                if cnt/total >= thr]

    @property
    def feature_frequencies(self): return self.freq_counter


# ─── FIX-1: STABLE PATHWAY PRESCREEN ─────────────────────────────────────────
# UNCHANGED — computes intersection across all folds before the CV loop
def compute_stable_pathway_prescreen(X_raw, y, cohort, genes, outer_cv, strat):
    log = get_logger()
    all_pathway_genes = set(g for gs in cfg.AXIS_DEFINITIONS.values() for g in gs)
    fold_sets = []
    for fid, (tr_idx, _) in enumerate(outer_cv.split(X_raw, strat)):
        Xtr = X_raw[tr_idx]; ctr = cohort[tr_idx]
        vm  = fit_variance_filter(Xtr, percentile=cfg.VAR_FILTER_PCT, cohort_train=ctr)
        gv  = [g for g, m in zip(genes, vm) if m]
        surviving = {g for g in gv if g in all_pathway_genes}
        fold_sets.append(surviving)
        log.info(f"  Stable prescreen fold {fid+1}: {len(surviving)} pathway genes")
    stable = sorted(set.intersection(*fold_sets))
    log.info(f"  Stable prescreen INTERSECTION: {len(stable)} genes")
    return stable

def build_pathway_prescreen(gv):
    all_pathway_genes = set(g for gs in cfg.AXIS_DEFINITIONS.values() for g in gs)
    gv_lookup = {g: i for i, g in enumerate(gv)}
    idx = sorted(gv_lookup[g] for g in all_pathway_genes if g in gv_lookup)
    return np.array(idx, dtype=int)


# ─── GENE PIPELINE EVAL HELPER ────────────────────────────────────────────────
def _gene_pipeline_eval(
    X_raw_tr, y_tr, c_tr,
    X_raw_te, y_te, c_te,
    genes, axis_set,
    stable_genes=None,   # list of gene names; None = per-fold variance-filtered prescreen
    log=None, label='', seed_base=None,
):
    """
    Full gene-level pipeline for one train→test split.
    Called twice per fold: (1) X_dev→X_val train eval, (2) X_tr→X_te test eval.

    Feature selection / dimensionality reduction is UNCHANGED from run_outer_fold:
      - variance filter (fit on tr), axis exclusion, global MI selection
      - archetype expert: bootstrap MI selection per archetype (ps_idx, top_local)
    Structural additions vs original:
      - PIPELINE_MODELS used for both baseline and experts
      - three calibration modes applied to output probabilities
      - baseline and routed results indexed [cal_mode][obj][mname][strat]

    Returns: (baseline, routed) — both [cal_mode][obj][m][strat]
    """
    if seed_base is None: seed_base = cfg.RANDOM_SEED

    if y_te.sum() == 0 or (1-y_te).sum() == 0:
        if log: log.warning(f"  [{label}] Single class in test — empty metrics returned")
        def _empty():
            return {cal: {obj: {m: {s: compute_metrics(y_te, np.full(len(y_te),0.5), 0.5)
                                    for s in cfg.THRESHOLD_STRATEGIES}
                                for m in PIPELINE_MODELS}
                          for obj in cfg.OPTUNA_OBJECTIVES}
                    for cal in CALIBRATION_MODES}
        return _empty(), _empty()

    # 1. Z-score (fit on tr only)
    np_ = fit_train_normalisation(X_raw_tr, c_tr)
    Xtrz = apply_train_normalisation(X_raw_tr, c_tr, np_)
    Xtez = apply_train_normalisation(X_raw_te, c_te, np_)

    # 2. Variance filter (fit on tr only)
    vm   = fit_variance_filter(X_raw_tr, percentile=cfg.VAR_FILTER_PCT, cohort_train=c_tr)
    Xtrv = Xtrz[:, vm]; Xtev = Xtez[:, vm]
    gv   = [g for g, m in zip(genes, vm) if m]

    # 3. Axis exclusion
    if cfg.EXCLUDE_AXIS_GENES_FROM_POOL:
        nm   = np.array([g not in axis_set for g in gv])
        Xtrf = Xtrv[:, nm]; Xtef = Xtev[:, nm]
        gf   = [g for g, m in zip(gv, nm) if m]
    else:
        Xtrf, Xtef, gf = Xtrv, Xtev, gv

    # 4. Routing (fit on tr only)
    atr, anames, _ = compute_axis_scores(Xtrz, genes)
    ate, _, _      = compute_axis_scores(Xtez, genes)
    router = ArchetypeRouter()
    router.fit(atr, anames)
    aprobtr = router.predict_proba(atr, anames)
    aprobte = router.predict_proba(ate, anames)
    ahtr    = aprobtr.argmax(axis=1)
    arch_labels = router.label_archetypes(atr, anames, y_tr)

    small = set()
    if log: log.info(f"  Archetypes [{label}]:")
    for k in range(router.k):
        nk   = int((ahtr==k).sum())
        flag = ' <- fallback' if nk < cfg.ARCHETYPE_MIN_TRAIN else ''
        if log:
            log.info(f"    A{k}({arch_labels[k]}): n={nk}  "
                     f"R={int(y_tr[ahtr==k].sum())}  "
                     f"resp={y_tr[ahtr==k].mean() if nk else 0:.2f}{flag}")
        if nk < cfg.ARCHETYPE_MIN_TRAIN: small.add(k)

    # 5. Global MI selection (fit on tr only)
    icv = StratifiedKFold(n_splits=cfg.N_INNER_FOLDS, shuffle=True, random_state=seed_base)
    sk  = min(cfg.SMOTE_K, int(y_tr.sum())-1, int((1-y_tr).sum())-1)
    mig = mutual_info_classif(Xtrf, y_tr, random_state=seed_base)
    tg  = np.argsort(mig)[-cfg.N_FEATURES_MI:]
    Xtrs = Xtrf[:, tg]; Xtes = Xtef[:, tg]

    # 6. Pathway prescreen (UNCHANGED logic — stable_genes or per-fold)
    gv_lookup         = {g: i for i, g in enumerate(gv)}
    all_pathway_genes = set(g for gs in cfg.AXIS_DEFINITIONS.values() for g in gs)
    if stable_genes is not None:
        pathway_ps_idx = np.array(
            [gv_lookup[g] for g in stable_genes if g in gv_lookup], dtype=int)
    else:
        pathway_ps_idx = np.array(
            sorted(gv_lookup[g] for g in all_pathway_genes if g in gv_lookup), dtype=int)
    ps             = {k: pathway_ps_idx for k in range(router.k)}
    X_pathway_full = Xtrv[:, pathway_ps_idx]

    # 7. Baseline: tune PIPELINE_MODELS on MI features, get raw OOF, fit calibrators
    raw_oof_bl      = {obj: {m: None for m in PIPELINE_MODELS} for obj in cfg.OPTUNA_OBJECTIVES}
    calibrators     = {obj: {m: {cal: None for cal in CALIBRATION_MODES} for m in PIPELINE_MODELS} for obj in cfg.OPTUNA_OBJECTIVES}
    oof_cal_bl      = {obj: {m: {cal: None for cal in CALIBRATION_MODES} for m in PIPELINE_MODELS} for obj in cfg.OPTUNA_OBJECTIVES}
    raw_te_bl       = {obj: {m: None for m in PIPELINE_MODELS} for obj in cfg.OPTUNA_OBJECTIVES}
    fallback_pipes  = {obj: {m: None for m in PIPELINE_MODELS} for obj in cfg.OPTUNA_OBJECTIVES}
    best_p_baseline = {obj: {m: {} for m in PIPELINE_MODELS} for obj in cfg.OPTUNA_OBJECTIVES}

    for obj in cfg.OPTUNA_OBJECTIVES:
        for mname in PIPELINE_MODELS:
            study = optuna.create_study(direction='maximize',
                sampler=optuna.samplers.TPESampler(seed=seed_base))
            study.optimize(make_objective(mname, Xtrs, y_tr, icv, sk, obj),
                           n_trials=cfg.N_OPTUNA_TRIALS, show_progress_bar=False)
            bp = study.best_params
            best_p_baseline[obj][mname] = bp
            oof_raw = get_oof(mname, Xtrs, y_tr, bp, icv, sk)
            raw_oof_bl[obj][mname] = oof_raw
            for cal_mode in CALIBRATION_MODES:
                c_obj, oof_c = fit_calibrator(oof_raw, y_tr, cal_mode)
                calibrators[obj][mname][cal_mode] = c_obj
                oof_cal_bl[obj][mname][cal_mode]  = oof_c
            Xfit, yfit = smote_data(Xtrs, y_tr, sk, seed_base)
            pipe_bl = Pipeline([('scl', StandardScaler()),
                                ('clf', build_classifier(mname, bp, yfit))])
            pipe_bl.fit(Xfit, yfit)
            fallback_pipes[obj][mname] = pipe_bl
            raw_te_bl[obj][mname] = pipe_bl.predict_proba(Xtes)[:, 1]

    # 8. Archetype expert tuning (pathway space) — UNCHANGED LR logic
    best_p_arch = {}
    for k in range(router.k):
        if k in small:
            best_p_arch[k] = best_p_baseline['MCC'].get('LR', {'C': 0.01})
            continue
        icv_arch = StratifiedKFold(n_splits=cfg.N_INNER_FOLDS, shuffle=True,
                                    random_state=seed_base+k)
        sk_arch  = min(cfg.SMOTE_K, int(y_tr.sum())-1, int((1-y_tr).sum())-1)
        study_arch = optuna.create_study(direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=seed_base+k))
        study_arch.optimize(
            make_archetype_objective(X_pathway_full, y_tr, icv_arch, sk_arch),
            n_trials=cfg.N_ARCHETYPE_OPTUNA_TRIALS, show_progress_bar=False)
        best_p_arch[k] = {'C': study_arch.best_params['C']}

    # 9. Train experts (PIPELINE_MODELS, UNCHANGED MI/bootstrap logic)
    mids    = {m: i for i, m in enumerate(PIPELINE_MODELS)}
    experts = {}
    for k in range(router.k):
        if k in small: continue
        for mname in PIPELINE_MODELS:
            # LR uses pathway-tuned C; EN uses pathway-tuned C + default l1_ratio;
            # RF/SVM use global baseline params — preserves existing behaviour for LR/RF
            if mname == 'LR':
                arch_params = best_p_arch[k]
            elif mname == 'EN':
                arch_params = {'C': best_p_arch[k]['C'], 'l1_ratio': 0.5}
            else:
                arch_params = best_p_baseline['MCC'][mname]
            exp = ArchetypeExpert(mname, model_id=mids[mname])
            exp.fit(Xtrv, y_tr, aprobtr[:, k], gv, arch_params,
                    ps_idx=ps[k], fold_id=seed_base % 10000, arch_id=k)
            experts[(k, mname)] = exp

    # 10. Collect gene data from trained experts and log stable features
    gene_data = {
        f'A{k}_{mname}': {
            'stable':      experts[(k, mname)].stable_features(),
            'frequencies': dict(experts[(k, mname)].feature_frequencies),
        }
        for k in range(router.k) for mname in PIPELINE_MODELS
        if (k, mname) in experts
    }
    if log:
        log.info(f"\n  [Expert gene data — {label}]")
        for am, gd in sorted(gene_data.items()):
            stable = gd['stable']
            if stable:
                log.info(f"    {am} stable (≥{cfg.ARCHETYPE_BOOTSTRAP_THRESHOLD:.0%}): "
                         f"{[f'{g}({f:.2f})' for g, f in stable[:8]]}")
            else:
                log.info(f"    {am}: no stable features above threshold")

    # 11. Raw routed predictions on test
    raw_te_routed = {obj: {m: None for m in PIPELINE_MODELS} for obj in cfg.OPTUNA_OBJECTIVES}
    for obj in cfg.OPTUNA_OBJECTIVES:
        for mname in PIPELINE_MODELS:
            prob_r = np.zeros(len(y_te)); ws = np.zeros(len(y_te))
            for k in range(router.k):
                aw = aprobte[:, k]
                if k in small or (k, mname) not in experts:
                    pk = fallback_pipes[obj][mname].predict_proba(Xtes)[:, 1]
                else:
                    pk = experts[(k, mname)].predict_proba_raw(Xtev, ps_idx=ps[k])
                prob_r += aw*pk; ws += aw
            ws = np.where(ws < 1e-8, 1.0, ws)
            raw_te_routed[obj][mname] = prob_r / ws

    # 11. Apply calibration + compute metrics
    def _blank():
        return {cal: {obj: {m: {} for m in PIPELINE_MODELS}
                      for obj in cfg.OPTUNA_OBJECTIVES}
                for cal in CALIBRATION_MODES}
    baseline = _blank(); routed = _blank()

    for cal_mode in CALIBRATION_MODES:
        for obj in cfg.OPTUNA_OBJECTIVES:
            for mname in PIPELINE_MODELS:
                cal_obj   = calibrators[obj][mname][cal_mode]
                oof_c     = oof_cal_bl[obj][mname][cal_mode]
                te_cal_bl = apply_calibrator(cal_obj, raw_te_bl[obj][mname], cal_mode)
                te_cal_r  = apply_calibrator(cal_obj, raw_te_routed[obj][mname], cal_mode)
                for strat in cfg.THRESHOLD_STRATEGIES:
                    t, _ = find_threshold(y_tr, oof_c, strat)
                    baseline[cal_mode][obj][mname][strat] = compute_metrics(y_te, te_cal_bl, t)
                    routed[cal_mode][obj][mname][strat]   = compute_metrics(y_te, te_cal_r,  t)

    return baseline, routed, gene_data


# ─── OUTER FOLD ───────────────────────────────────────────────────────────────
def run_outer_fold(fold_id, tr_idx, te_idx, X_raw, y, cohort, genes, axis_set,
                    stable_pathway_genes=None):
    log = get_logger(); t0 = time.time(); sep = '─'*62
    log.info(f"\n{sep}\nOUTER FOLD {fold_id+1}/{cfg.N_OUTER_FOLDS}  "
             f"(train={len(tr_idx)}, test={len(te_idx)})\n{sep}")

    Xtr, Xte = X_raw[tr_idx], X_raw[te_idx]
    ytr, yte = y[tr_idx],     y[te_idx]
    ctr, cte = cohort[tr_idx], cohort[te_idx]
    log.info(f"Train {dict(zip(*np.unique(ctr, return_counts=True)))}  "
             f"{int(ytr.sum())}R/{int((1-ytr).sum())}NR")

    # ── Train holdout split — reserved BEFORE any model training ─────────────
    # dev (80%) → all tuning and training; val (20%) → unbiased train eval
    cohort_map  = {c: i for i, c in enumerate(np.unique(ctr))}
    strat_tr    = ytr.astype(int)*10 + np.array([cohort_map[c] for c in ctr])
    dev_idx, val_idx = _tts(
        np.arange(len(ytr)), test_size=TRAIN_HOLDOUT_FRAC,
        stratify=strat_tr, random_state=cfg.RANDOM_SEED + fold_id*100)

    log.info(f"Train holdout: dev n={len(dev_idx)} (R={int(ytr[dev_idx].sum())}) | "
             f"val n={len(val_idx)} (R={int(ytr[val_idx].sum())})")

    # Stable prescreen gene names for the test eval call
    stable_for_test = stable_pathway_genes  # None = per-fold prescreen

    # ── Pipeline call 1: dev → val (unbiased within-training eval) ───────────
    log.info("\n[Gene pipeline: dev → val  (train eval)]")
    baseline_tr, routed_tr, gene_data_tr = _gene_pipeline_eval(
        Xtr[dev_idx], ytr[dev_idx], ctr[dev_idx],
        Xtr[val_idx], ytr[val_idx], ctr[val_idx],
        genes, axis_set,
        stable_genes=None,   # per-dev prescreen for train eval
        log=log, label=f'train_fold{fold_id+1}',
        seed_base=cfg.RANDOM_SEED + 50000 + fold_id*1000,
    )

    # ── Pipeline call 2: full outer train → outer test ────────────────────────
    log.info("\n[Gene pipeline: full_train → test  (test eval)]")
    baseline_te, routed_te, gene_data_te = _gene_pipeline_eval(
        Xtr, ytr, ctr,
        Xte, yte, cte,
        genes, axis_set,
        stable_genes=stable_for_test,
        log=log, label=f'test_fold{fold_id+1}',
        seed_base=cfg.RANDOM_SEED + fold_id*1000,
    )

    # ── Log fold results ──────────────────────────────────────────────────────
    for cal_mode in CALIBRATION_MODES:
        log.info(f"\n  ═══ Fold {fold_id+1} | Calibration: {cal_mode} ═══")
        for obj in cfg.OPTUNA_OBJECTIVES:
            log.info(f"  [{obj}]")
            for mname in PIPELINE_MODELS:
                log.info(f"    --- {mname} ---")
                for strat in cfg.THRESHOLD_STRATEGIES:
                    for lbl, d in [
                        ('TEST  ROUTED  ', routed_te),
                        ('TEST  BASELINE', baseline_te),
                        ('TRAIN ROUTED  ', routed_tr),
                        ('TRAIN BASELINE', baseline_tr),
                    ]:
                        m = d[cal_mode][obj][mname][strat]
                        log.info(
                            f"    [{lbl}] {mname}[{strat} t={m['Threshold']:.2f}]: "
                            f"AUC={m['AUC']:.4f}  MCC={m['MCC']:.4f}  "
                            f"Sens={m['Sensitivity']:.2f}  Spec={m['Specificity']:.2f}  "
                            f"TP={m['TP']} TN={m['TN']} FP={m['FP']} FN={m['FN']}")

    elapsed = time.time()-t0
    log.info(f"\nFold {fold_id+1} done in {elapsed:.0f}s ({elapsed/60:.1f}min)")

    result = {
        'fold_id':      fold_id,
        'routed_te':    routed_te,    'baseline_te':  baseline_te,
        'routed_tr':    routed_tr,    'baseline_tr':  baseline_tr,
        'gene_data_te': gene_data_te, # experts trained on full fold train — use for discovery
        'gene_data_tr': gene_data_tr, # experts trained on dev split only — smaller N
        'true_test':    yte.tolist(), 'test_idx':     te_idx.tolist(),
        'n_test':       int(len(yte)), 'n_R_test':    int(yte.sum()),
        'n_dev':        int(len(dev_idx)), 'n_val':   int(len(val_idx)),
        'n_R_dev':      int(ytr[dev_idx].sum()),
        'n_R_val':      int(ytr[val_idx].sum()),
        'time_seconds': elapsed,
    }
    cfg.OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    ckpt_path = cfg.OUTPUT_DIR / f'gene_pipeline_fold{fold_id}.pkl'
    with open(ckpt_path, 'wb') as f: pickle.dump(result, f)
    log.info(f"Fold {fold_id+1} checkpoint → {ckpt_path}")
    return result


# ─── POST-PROCESSING ──────────────────────────────────────────────────────────
# UNCHANGED
def bootstrap_ci(y_true, y_prob, threshold=None, n=None):
    n   = n or cfg.N_BOOTSTRAP_CI
    log = get_logger()
    log.info(f"\nBootstrap CI ({n} resamples, "
             f"threshold={f'{threshold:.3f}' if threshold else 'per-resample'})...")
    rng = np.random.RandomState(cfg.RANDOM_SEED)
    metrics = defaultdict(list)
    for _ in range(n):
        idx = rng.choice(len(y_true), size=len(y_true), replace=True)
        ys, ps = y_true[idx], y_prob[idx]
        if ys.sum()==0 or ys.sum()==len(ys): continue
        t = threshold if threshold is not None else find_threshold(ys, ps, 'MCC')[0]
        m = compute_metrics(ys, ps, t)
        for k, v in m.items():
            if isinstance(v, float): metrics[k].append(v)
    ci = {}
    for k, vals in metrics.items():
        a = np.array(vals)
        ci[k] = {'mean':float(a.mean()), 'lo':float(np.percentile(a,2.5)),
                  'hi':float(np.percentile(a,97.5))}
    for k in ['AUC','MCC','Sensitivity','Specificity','Brier']:
        if k in ci:
            log.info(f"  {k}: {ci[k]['mean']:.4f} [{ci[k]['lo']:.4f}–{ci[k]['hi']:.4f}]")
    return ci

def permutation_test(y_true, y_prob, threshold=None, n=None):
    n   = n or cfg.N_PERMUTATIONS
    log = get_logger()
    t_real   = threshold if threshold is not None else find_threshold(y_true, y_prob, 'MCC')[0]
    real_mcc = float(compute_metrics(y_true, y_prob, t_real)['MCC'])
    try:    real_auc = float(roc_auc_score(y_true, y_prob))
    except: real_auc = 0.5
    log.info(f"\nPermutation test ({n} shuffles, threshold={t_real:.3f})...")
    log.info(f"  Real MCC={real_mcc:.4f}  Real AUC={real_auc:.4f}")
    rng   = np.random.RandomState(cfg.RANDOM_SEED)
    pmccs = []; paucs = []
    for _ in range(n):
        yp = rng.permutation(y_true)
        pmccs.append(compute_metrics(yp, y_prob, t_real)['MCC'])
        try:    paucs.append(float(roc_auc_score(yp, y_prob)))
        except: paucs.append(0.5)
    p_mcc = float(np.mean(np.array(pmccs) >= real_mcc))
    p_auc = float(np.mean(np.array(paucs) >= real_auc))
    log.info(f"  p(MCC)={p_mcc:.4f}  p(AUC)={p_auc:.4f}  "
             f"({'significant' if p_mcc<0.05 else 'not significant'} alpha=0.05)")
    return {'real_mcc':real_mcc, 'real_auc':real_auc,
            'p_mcc':p_mcc, 'p_auc':p_auc,
            'perm_mccs':pmccs, 'perm_aucs':paucs}


# ─── MAIN ─────────────────────────────────────────────────────────────────────
def run_full_pipeline(folds_to_run=None):
    """
    folds_to_run : list of 0-based fold indices to run, e.g. [2, 3].
                   If None, all folds are run.
                   Others loaded from per-fold checkpoints if available.

    CLI: python pipeline.py 2 3   (runs folds 2 and 3 only)
    """
    log = setup_logging(); t0 = time.time()
    axis_genes = sorted(set(g for gs in cfg.AXIS_DEFINITIONS.values() for g in gs))
    axis_set   = set(axis_genes)

    banner = [
        "="*62,
        " BIOADAPT-ARCHETYPE v6 — GENE PIPELINE (structural v2)",
        f" Cohorts      : {cfg.PRIMARY_COHORTS}",
        f" Models       : {PIPELINE_MODELS}",
        f" Calibration  : {CALIBRATION_MODES}",
        f" FIX-1        : F2/F3 min Spec >= {F2F3_MIN_SPEC}",
        f" Train holdout: {int(TRAIN_HOLDOUT_FRAC*100)}% per fold (unbiased train eval)",
        f" CV           : {cfg.N_OUTER_FOLDS}x{cfg.N_INNER_FOLDS}",
        f" Bootstrap    : {cfg.N_BOOTSTRAP} rounds/expert (MI selection UNCHANGED)",
        "="*62,
    ]
    for l in banner: log.info(l)

    X, y, cohort, genes, norm = load_pooled()
    log.info(f"Loaded: {X.shape[0]}x{X.shape[1]}  R={int(y.sum())}/{len(y)}")
    for cid in np.unique(cohort):
        m = cohort==cid
        log.info(f"  {cid}: n={m.sum()}  R={int(y[m].sum())}  resp={y[m].mean():.2f}")

    cohort_idx   = {c: i for i, c in enumerate(np.unique(cohort))}
    strat        = y*100 + np.array([cohort_idx[c] for c in cohort])
    outer_cv     = StratifiedKFold(n_splits=cfg.N_OUTER_FOLDS, shuffle=True,
                                    random_state=cfg.RANDOM_SEED)

    # FIX-1: stable prescreen computed once before the loop (UNCHANGED)
    stable_pathway_genes = None
    if cfg.USE_STABLE_PRESCREEN:
        log.info("\n[Computing stable pathway prescreen...]")
        stable_pathway_genes = compute_stable_pathway_prescreen(
            X, y, cohort, genes, outer_cv, strat)
        log.info(f"Stable prescreen: {len(stable_pathway_genes)} genes\n")

    all_splits  = list(enumerate(outer_cv.split(X, strat)))
    fold_results = {}

    for fid, (tr, te) in all_splits:
        ckpt_path = cfg.OUTPUT_DIR / f'gene_pipeline_fold{fid}.pkl'
        if folds_to_run is not None and fid not in folds_to_run:
            if ckpt_path.exists():
                with open(ckpt_path, 'rb') as f:
                    fold_results[fid] = pickle.load(f)
                log.info(f"Fold {fid+1}: loaded from checkpoint")
            else:
                log.warning(f"Fold {fid+1}: not in folds_to_run, no checkpoint — excluded")
        else:
            fold_results[fid] = run_outer_fold(
                fid, tr, te, X, y, cohort, genes, axis_set,
                stable_pathway_genes=stable_pathway_genes)

    fold_results_list = [fold_results[fid] for fid in sorted(fold_results)]

    # ── Aggregate ─────────────────────────────────────────────────────────────
    rows = []
    for res in fold_results_list:
        fid = res['fold_id']
        for cal_mode in CALIBRATION_MODES:
            for obj in cfg.OPTUNA_OBJECTIVES:
                for mname in PIPELINE_MODELS:
                    for thr_strat in cfg.THRESHOLD_STRATEGIES:
                        base = {'fold_id':fid, 'calibration':cal_mode,
                                'objective':obj, 'model':mname,
                                'threshold_strategy':thr_strat,
                                'n_test':res['n_test'], 'n_R_test':res['n_R_test'],
                                'n_dev':res['n_dev'],   'n_val':res['n_val']}
                        rows.append({**base, 'type':'Test_Routed',
                                     **res['routed_te'][cal_mode][obj][mname][thr_strat]})
                        rows.append({**base, 'type':'Test_Baseline',
                                     **res['baseline_te'][cal_mode][obj][mname][thr_strat]})
                        rows.append({**base, 'type':'Train_Routed_HoldOut',
                                     **res['routed_tr'][cal_mode][obj][mname][thr_strat]})
                        rows.append({**base, 'type':'Train_Baseline_HoldOut',
                                     **res['baseline_tr'][cal_mode][obj][mname][thr_strat]})

    df = pd.DataFrame(rows)

    log.info(f"\n{'='*70}")
    log.info("AGGREGATE SUMMARY — MCC obj | MCC threshold | isotonic_on")
    log.info(f"{'='*70}")
    sub = df[(df.objective=='MCC')&(df.threshold_strategy=='MCC')&(df.calibration=='isotonic_on')]
    for dtype in ['Test_Routed','Test_Baseline','Train_Routed_HoldOut','Train_Baseline_HoldOut']:
        log.info(f"\n  [{dtype}]")
        for mname in PIPELINE_MODELS:
            d = sub[(sub.type==dtype)&(sub.model==mname)]
            if d.empty: continue
            log.info(f"    {mname}: AUC={d['AUC'].mean():.4f}±{d['AUC'].std():.4f}  "
                     f"MCC={d['MCC'].mean():.4f}±{d['MCC'].std():.4f}  "
                     f"Sens={d['Sensitivity'].mean():.3f}  Spec={d['Specificity'].mean():.3f}")

    # Pool OOF for bootstrap CI and permutation test (using LR/MCC/isotonic_on as reference)
    oof_prob = np.zeros(len(y)); oof_true = np.zeros(len(y), dtype=int)
    for fold in fold_results_list:
        ti = fold['test_idx']
        # Use isotonic_on calibrated routed LR/MCC predictions
        key_cal = 'isotonic_on'; key_obj = 'MCC'; key_m = 'LR'
        oof_prob[ti] = np.array(
            [fold['routed_te'][key_cal][key_obj][key_m][s]['AUC']
             for s in ['MCC']][0:1] or [0.5])
        oof_true[ti] = fold['true_test']

    ref_thresholds = [fold['routed_te']['isotonic_on']['MCC']['LR']['SENS_CONSTRAINED']['Threshold']
                      for fold in fold_results_list]
    mean_fold_threshold = float(np.mean(ref_thresholds))
    log.info(f"\nMean fold SENS_CONSTRAINED threshold (LR/MCC/isotonic): {mean_fold_threshold:.3f}")

    # ── Discovery gene aggregation (from test-eval experts — full fold train) ──
    # gene_data_te is used because those experts were trained on the complete
    # outer-fold training set, giving the fullest bootstrap coverage.
    log.info(f"\n{'─'*62}\nDISCOVERY GENES (aggregated across {len(fold_results_list)} folds)\n{'─'*62}")
    disc = defaultdict(Counter)
    wf   = defaultdict(list)
    for fold in fold_results_list:
        for am, gd in fold.get('gene_data_te', {}).items():
            for g, f in gd['stable']:
                disc[am][g] += 1
            wf[am].append(gd['frequencies'])

    disc_rows = []
    for am, ctr_d in sorted(disc.items()):
        top_str = [f"{g}({n}/{len(fold_results_list)})" for g, n in ctr_d.most_common(10)]
        log.info(f"  {am}: {top_str}")
        for g, fc in ctr_d.most_common():
            bfreqs = [fd.get(g, 0) / cfg.N_BOOTSTRAP for fd in wf[am] if g in fd]
            disc_rows.append({
                'archetype_model':        am,
                'gene':                   g,
                'fold_frequency':         fc,
                'n_folds':                len(fold_results_list),
                'mean_bootstrap_freq':    float(np.mean(bfreqs)) if bfreqs else 0.0,
            })

    cfg.OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    df.to_csv(cfg.OUTPUT_DIR / 'gene_pipeline_v2_metrics.csv', index=False)
    pd.DataFrame(disc_rows).to_csv(cfg.OUTPUT_DIR / 'gene_pipeline_v2_discovery.csv', index=False)
    with open(cfg.OUTPUT_DIR / 'gene_pipeline_v2_results.pkl', 'wb') as f:
        pickle.dump(fold_results_list, f)
    log.info(f"\nSaved to output/gene_pipeline_v2_metrics.csv")
    log.info(f"Saved to output/gene_pipeline_v2_discovery.csv")
    log.info(f"Total: {(time.time()-t0)/60:.1f} min")
    return fold_results_list


if __name__ == '__main__':
    import sys as _sys
    folds = [int(x) for x in _sys.argv[1:]] if len(_sys.argv) > 1 else None
    run_full_pipeline(folds_to_run=folds)