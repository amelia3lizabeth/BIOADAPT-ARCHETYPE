"""
BIOADAPT-ARCHETYPE v6: LOCO Validation — PATHWAY AGGREGATION (v5)
=================================================================
FIX-1       : find_threshold F2/F3 enforces min Spec >= 0.10.
CALIBRATION : isotonic_on | isotonic_off | platt — all reported.
MODELS      : LR (L1), RF, SVM, EN (Elastic Net) — routed + baseline.
FIX-ARCH-TUNE: Membership-weighted archetype tuning per model.

TRAIN EVAL (unbiased):
  A stratified 20 pct hold-out of the LOCO training set is reserved
  BEFORE any training begins (matching the consensus pipeline pattern).
  All tuning, calibration, and expert training runs on the remaining
  80 pct (X_dev). The held-out 20 pct (X_val) is evaluated after.
  The full training set is then used separately to train the final model
  evaluated on the held-out cohort.
  CSV type labels:
    Test_Routed / Test_Baseline         — held-out cohort evaluation
    Train_Routed_HoldOut / Train_Baseline_HoldOut — within-training unbiased eval
"""
import sys, time, logging, pickle, warnings
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression as _LogReg
from sklearn.model_selection import train_test_split as _tts
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (roc_auc_score, matthews_corrcoef,
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
                          fit_train_normalisation, apply_train_normalisation)
from archetype_router import ArchetypeRouter

PIPELINE_MODELS   = ['LR', 'RF', 'SVM', 'EN']
CALIBRATION_MODES = ['isotonic_on', 'isotonic_off', 'platt']
F2F3_MIN_SPEC     = 0.10
TRAIN_HOLDOUT_FRAC = 0.20   # fraction of LOCO train set withheld for train eval


# ─── LOGGING ─────────────────────────────────────────────────────────────────
def setup_logging(label='loco_pathway_v5'):
    cfg.OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    log = logging.getLogger('bioadapt_loco_pw_v5')
    log.setLevel(logging.DEBUG)
    log.handlers.clear()
    fh = logging.FileHandler(cfg.OUTPUT_DIR / f'run_{label}.log', mode='w')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s', '%Y-%m-%d %H:%M:%S'))
    log.addHandler(fh)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter('%(message)s'))
    log.addHandler(ch)
    return log


# ─── PATHWAY AGGREGATION ─────────────────────────────────────────────────────
def get_pathway_scores(X_zscored, all_genes, axis_definitions):
    gene_to_idx   = {g: i for i, g in enumerate(all_genes)}
    pathway_names = list(axis_definitions.keys())
    X_scores = np.zeros((X_zscored.shape[0], len(pathway_names)))
    for i, pname in enumerate(pathway_names):
        p_genes = [g for g in axis_definitions[pname] if g in gene_to_idx]
        if not p_genes: continue
        p_idx = [gene_to_idx[g] for g in p_genes]
        X_scores[:, i] = np.mean(X_zscored[:, p_idx], axis=1)
    return X_scores, pathway_names


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
    return {'AUC':auc, 'MCC':float(matthews_corrcoef(y,pred)),
            'Sensitivity':float(sens), 'Specificity':float(spec),
            'Brier':float(brier_score_loss(y,prob)), 'Threshold':float(threshold),
            'TP':int(tp),'TN':int(tn),'FP':int(fp),'FN':int(fn)}


# ─── CALIBRATION ─────────────────────────────────────────────────────────────
def fit_calibrator(oof_raw, y_tr, mode):
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
            C=params['C'], penalty='l1', solver='liblinear',
            max_iter=1000, class_weight='balanced',
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
            max_features=params.get('max_features', 1.0), class_weight='balanced',
            random_state=cfg.RANDOM_SEED, n_jobs=cfg.N_JOBS)
    elif mname == 'SVM':
        return SVC(
            C=params['C'], kernel=params.get('kernel', 'rbf'),
            gamma='scale', probability=True, class_weight='balanced',
            random_state=cfg.RANDOM_SEED)
    else:
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
                C=trial.suggest_float('C', 1e-3, 5.0, log=True),
                penalty='l1', solver='liblinear', max_iter=1000,
                class_weight='balanced', random_state=cfg.RANDOM_SEED)
        elif mname == 'EN':
            clf = LogisticRegression(
                C=trial.suggest_float('C', 1e-3, 5.0, log=True),
                l1_ratio=trial.suggest_float('l1_ratio', 0.1, 0.9),
                penalty='elasticnet', solver='saga', max_iter=2000,
                class_weight='balanced', random_state=cfg.RANDOM_SEED)
        elif mname == 'RF':
            clf = RandomForestClassifier(
                n_estimators=trial.suggest_int('n_estimators', 50, 300),
                max_depth=trial.suggest_int('max_depth', 2, 6),
                max_features=trial.suggest_float('max_features', 0.3, 1.0),
                class_weight='balanced',
                random_state=cfg.RANDOM_SEED, n_jobs=cfg.N_JOBS)
        elif mname == 'SVM':
            clf = SVC(
                C=trial.suggest_float('C', 1e-2, 10.0, log=True),
                kernel=trial.suggest_categorical('kernel', ['linear', 'rbf']),
                gamma='scale', probability=True, class_weight='balanced',
                random_state=cfg.RANDOM_SEED)
        else:
            spw = float((y_tr==0).sum())/max((y_tr==1).sum(),1)
            clf = xgb.XGBClassifier(
                learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                n_estimators=trial.suggest_int('n_estimators', 50, 200),
                max_depth=trial.suggest_int('max_depth', 2, 4),
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
    oof = np.zeros(len(y_tr))
    for tri, vai in inner_cv.split(X_tr, y_tr):
        Xin, yin = smote_data(X_tr[tri], y_tr[tri], smote_k, cfg.RANDOM_SEED)
        clf  = build_classifier(mname, params, yin)
        pipe = Pipeline([('scl', StandardScaler()), ('clf', clf)])
        pipe.fit(Xin, yin)
        oof[vai] = pipe.predict_proba(X_tr[vai])[:, 1]
    return oof


# ─── FIX-ARCH-TUNE ────────────────────────────────────────────────────────────
def make_weighted_archetype_sample(X_pathway, y, mem, arch_seed):
    w_b = mem / mem.sum()
    rng = np.random.RandomState(arch_seed)
    idx = rng.choice(len(y), size=int(len(y)*cfg.BOOTSTRAP_FRAC), replace=True, p=w_b)
    return X_pathway[idx], y[idx]


# ─── ARCHETYPE EXPERT ─────────────────────────────────────────────────────────
class ArchetypeExpert:
    def __init__(self, mname, model_id=0):
        self.mname            = mname
        self.model_id         = model_id
        self.bootstrap_models = []
        self.freq_counter     = Counter()

    def fit(self, X_path, y, mem, pathway_names, params, fold_id=0, arch_id=0):
        base = cfg.RANDOM_SEED + fold_id*10000 + arch_id*1000 + self.model_id*100
        rng  = np.random.RandomState(base)
        for b in range(cfg.N_BOOTSTRAP):
            n_samp = int(len(y)*cfg.BOOTSTRAP_FRAC)
            w_b    = mem / mem.sum()
            idx    = rng.choice(len(y), size=n_samp, replace=True, p=w_b)
            Xb, yb = X_path[idx], y[idx]
            if yb.sum() < 2 or (1-yb).sum() < 2: continue
            Xbs, ybs = smote_data(Xb, yb, cfg.SMOTE_K, base+b)
            clf  = build_classifier(self.mname, params, ybs)
            pipe = Pipeline([('scl', StandardScaler()), ('clf', clf)])
            try:
                pipe.fit(Xbs, ybs)
                self.bootstrap_models.append(pipe)
                if self.mname in ('LR', 'EN'):
                    coeffs = pipe.named_steps['clf'].coef_[0]
                    for i, c in enumerate(coeffs):
                        if abs(c) > 1e-5: self.freq_counter[pathway_names[i]] += 1
                else:
                    for pname in pathway_names: self.freq_counter[pname] += 1
            except: pass
        return self

    def predict_proba_raw(self, X_path):
        if not self.bootstrap_models: return np.full(X_path.shape[0], 0.5)
        probs = []
        for pipe in self.bootstrap_models:
            try: probs.append(pipe.predict_proba(X_path)[:, 1])
            except: continue
        return np.mean(probs, axis=0) if probs else np.full(X_path.shape[0], 0.5)

    def stable_features(self, thr=None):
        if thr is None: thr = cfg.ARCHETYPE_BOOTSTRAP_THRESHOLD
        total = len(self.bootstrap_models)
        if total == 0: return []
        return [(g, cnt/total) for g, cnt in self.freq_counter.most_common()
                if cnt/total >= thr]


# ─── PIPELINE EVAL HELPER ─────────────────────────────────────────────────────
def _pipeline_eval(
    X_pw_tr, y_tr,
    axis_tr, anames, pathway_names,
    X_pw_te, y_te, axis_te,
    log=None, label='', seed_base=None,
):
    """
    Full pipeline for one train→test split. Called twice per LOCO fold:
      (1) X_dev → X_val   (unbiased within-training evaluation)
      (2) X_pathway_tr → X_pathway_te  (held-out cohort evaluation)

    Inputs:
      X_pw_tr / y_tr   — pathway scores and labels for training
      axis_tr / anames — pre-computed axis scores for training (used by router)
      pathway_names    — pathway name list for expert feature tracking
      X_pw_te / y_te   — pathway scores and labels for test
      axis_te          — pre-computed axis scores for test patients
      seed_base        — base random seed (differ between the two calls)

    Returns:
      baseline[cal][obj][m][strat]
      routed  [cal][obj][m][strat]
    """
    if seed_base is None:
        seed_base = cfg.RANDOM_SEED

    # Single-class guard
    if y_te.sum() == 0 or (1-y_te).sum() == 0:
        if log: log.warning(f"  [{label}] Single class in test — skipping eval")
        def _empty():
            return {cal: {obj: {m: {s: compute_metrics(y_te, np.full(len(y_te), 0.5), 0.5)
                                    for s in cfg.THRESHOLD_STRATEGIES}
                                for m in PIPELINE_MODELS}
                          for obj in cfg.OPTUNA_OBJECTIVES}
                    for cal in CALIBRATION_MODES}
        return _empty(), _empty()

    # ── Router (fit on train only) ─────────────────────────────────────────
    router = ArchetypeRouter()
    router.fit(axis_tr, anames)
    aprobtr = router.predict_proba(axis_tr, anames)
    aprobte = router.predict_proba(axis_te, anames)
    ahtr    = aprobtr.argmax(axis=1)
    arch_labels = router.label_archetypes(axis_tr, anames, y_tr)

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

    icv = StratifiedKFold(n_splits=cfg.N_INNER_FOLDS, shuffle=True,
                          random_state=seed_base)
    sk  = min(cfg.SMOTE_K, int(y_tr.sum())-1, int((1-y_tr).sum())-1)

    # ── Baseline: tune, OOF, calibrate ────────────────────────────────────
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
            study.optimize(make_objective(mname, X_pw_tr, y_tr, icv, sk, obj),
                           n_trials=cfg.N_OPTUNA_TRIALS, show_progress_bar=False)
            bp = study.best_params
            best_p_baseline[obj][mname] = bp
            oof_raw = get_oof(mname, X_pw_tr, y_tr, bp, icv, sk)
            raw_oof_bl[obj][mname] = oof_raw
            for cal_mode in CALIBRATION_MODES:
                c_obj, oof_c = fit_calibrator(oof_raw, y_tr, cal_mode)
                calibrators[obj][mname][cal_mode] = c_obj
                oof_cal_bl[obj][mname][cal_mode]  = oof_c
            Xfit, yfit = smote_data(X_pw_tr, y_tr, sk, seed_base)
            pipe_bl = Pipeline([('scl', StandardScaler()),
                                ('clf', build_classifier(mname, bp, yfit))])
            pipe_bl.fit(Xfit, yfit)
            fallback_pipes[obj][mname] = pipe_bl
            raw_te_bl[obj][mname] = pipe_bl.predict_proba(X_pw_te)[:, 1]

    # ── FIX-ARCH-TUNE ─────────────────────────────────────────────────────
    best_p_arch = {obj: {k: {m: {} for m in PIPELINE_MODELS}
                          for k in range(router.k)}
                   for obj in cfg.OPTUNA_OBJECTIVES}
    for obj in cfg.OPTUNA_OBJECTIVES:
        for k in range(router.k):
            if k in small:
                for mname in PIPELINE_MODELS:
                    best_p_arch[obj][k][mname] = best_p_baseline[obj][mname].copy()
                continue
            arch_seed = seed_base + k*100 + cfg.OPTUNA_OBJECTIVES.index(obj)*10
            X_w, y_w  = make_weighted_archetype_sample(
                X_pw_tr, y_tr, aprobtr[:, k], arch_seed)
            if y_w.sum() < 2 or (1-y_w).sum() < 2:
                for mname in PIPELINE_MODELS:
                    best_p_arch[obj][k][mname] = best_p_baseline[obj][mname].copy()
                continue
            for mname in PIPELINE_MODELS:
                sk_arch = min(cfg.SMOTE_K, int(y_w.sum())-1, int((1-y_w).sum())-1)
                if sk_arch < 1:
                    best_p_arch[obj][k][mname] = best_p_baseline[obj][mname].copy()
                    continue
                icv_arch   = StratifiedKFold(n_splits=cfg.N_INNER_FOLDS, shuffle=True,
                                             random_state=seed_base+k)
                model_seed = arch_seed + PIPELINE_MODELS.index(mname)*3
                st = optuna.create_study(direction='maximize',
                    sampler=optuna.samplers.TPESampler(seed=model_seed))
                st.optimize(make_objective(mname, X_w, y_w, icv_arch, sk_arch, obj),
                            n_trials=cfg.N_ARCHETYPE_OPTUNA_TRIALS,
                            show_progress_bar=False)
                best_p_arch[obj][k][mname] = st.best_params

    # ── Train experts ──────────────────────────────────────────────────────
    mids    = {m: i for i, m in enumerate(PIPELINE_MODELS)}
    experts = {}
    for obj in cfg.OPTUNA_OBJECTIVES:
        for k in range(router.k):
            if k in small: continue
            for mname in PIPELINE_MODELS:
                exp = ArchetypeExpert(mname, model_id=mids[mname])
                exp.fit(X_pw_tr, y_tr, aprobtr[:, k],
                        pathway_names, best_p_arch[obj][k][mname],
                        fold_id=seed_base % 10000, arch_id=k)
                experts[(obj, k, mname)] = exp
                if obj == 'MCC' and mname in ('LR', 'EN') and log:
                    stable = exp.stable_features()
                    if stable:
                        log.info(f"    A{k}/{mname} [{label}] Stable: "
                                 f"{[p for p,_ in stable[:5]]}")

    # ── Raw routed predictions on test ─────────────────────────────────────
    raw_te_routed = {obj: {m: None for m in PIPELINE_MODELS} for obj in cfg.OPTUNA_OBJECTIVES}
    for obj in cfg.OPTUNA_OBJECTIVES:
        for mname in PIPELINE_MODELS:
            prob_te = np.zeros(len(y_te)); ws_te = np.zeros(len(y_te))
            for k in range(router.k):
                aw = aprobte[:, k]
                pk = (fallback_pipes[obj][mname].predict_proba(X_pw_te)[:, 1]
                      if k in small or (obj, k, mname) not in experts
                      else experts[(obj, k, mname)].predict_proba_raw(X_pw_te))
                prob_te += aw*pk; ws_te += aw
            ws_te = np.where(ws_te < 1e-8, 1.0, ws_te)
            raw_te_routed[obj][mname] = prob_te / ws_te

    # ── Apply calibration + compute metrics ────────────────────────────────
    def _blank():
        return {cal: {obj: {m: {} for m in PIPELINE_MODELS}
                      for obj in cfg.OPTUNA_OBJECTIVES}
                for cal in CALIBRATION_MODES}
    baseline = _blank()
    routed   = _blank()

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

    return baseline, routed


# ─── LOCO PIPELINE ───────────────────────────────────────────────────────────
def run_loco():
    log = setup_logging()
    t0  = time.time()
    log.info("="*62)
    log.info(" BIOADAPT-ARCHETYPE v6 — LOCO PATHWAY (v5)")
    log.info(f" Cohorts       : {cfg.LOCO_COHORTS}")
    log.info(f" Models        : {PIPELINE_MODELS}")
    log.info(f" Calibration   : {CALIBRATION_MODES}")
    log.info(f" FIX-1         : F2/F3 min Spec >= {F2F3_MIN_SPEC}")
    log.info(f" Train holdout : {int(TRAIN_HOLDOUT_FRAC*100)}% of LOCO train set")
    log.info("="*62)

    X, y, cohort, genes, _ = load_pooled(cfg.LOCO_COHORTS, common_genes_only=True)
    log.info(f"Loaded {X.shape[0]} patients | {X.shape[1]} genes")
    for cid in np.unique(cohort):
        m = cohort==cid
        log.info(f"  {cid}: n={m.sum()}  R={int(y[m].sum())}  resp={y[m].mean():.2f}")

    all_results = []

    for held_out in cfg.LOCO_COHORTS:
        t_fold = time.time()
        log.info(f"\n{'─'*62}\nHELD-OUT: {held_out}\n{'─'*62}")

        mho=cohort==held_out; mtr=~mho
        Xtr,Xte = X[mtr],X[mho]
        ytr,yte = y[mtr],y[mho]
        ctr,cte = cohort[mtr],cohort[mho]

        log.info(f"Train cohorts: {dict(zip(*np.unique(ctr,return_counts=True)))}")
        log.info(f"Test  cohort : n={len(yte)}  R={int(yte.sum())}  NR={int((yte==0).sum())}")
        if yte.sum()==0 or (1-yte).sum()==0:
            log.warning("Skipping: single class in test"); continue

        # ── Z-score (fit on full LOCO train, applied once) ──
        np_ = fit_train_normalisation(Xtr, ctr)
        Xtrz = apply_train_normalisation(Xtr, ctr, np_)
        Xtez = apply_train_normalisation(Xte, cte, np_)

        # ── Pathway aggregation and axis scores (computed once) ──
        X_pathway_tr, pathway_names = get_pathway_scores(Xtrz, genes, cfg.AXIS_DEFINITIONS)
        X_pathway_te, _             = get_pathway_scores(Xtez, genes, cfg.AXIS_DEFINITIONS)
        atr, anames, _ = compute_axis_scores(Xtrz, genes)
        ate, _, _      = compute_axis_scores(Xtez, genes)
        log.info(f"Aggregated {len(genes)} genes → {len(pathway_names)} pathways.")

        # ── Cohort-archetype confound check (informational, full training set) ──
        router_check = ArchetypeRouter()
        router_check.fit(atr, anames)
        ahtr_check = router_check.predict_proba(atr, anames).argmax(axis=1)
        confound = False
        for cid in np.unique(ctr):
            cmask = ctr==cid; row = []
            for k in range(router_check.k):
                frac = float((cmask&(ahtr_check==k)).sum())/cmask.sum()
                row.append(f"A{k}={frac:.2f}")
                if frac >= 0.80: confound = True
            log.info(f"  Composition {cid}: {' | '.join(row)}")
        log.info("  ⚠ Potential cohort-archetype confound." if confound
                 else "  ✓ No strong cohort-archetype confound.")

        # ── TRAIN EVAL: stratified holdout of 20% of LOCO train set ──────────
        # This partition is decided BEFORE any model training.
        # dev = 80% of train — all tuning/training happens here
        # val = 20% of train — evaluated after, never seen during training
        cohort_map  = {c: i for i, c in enumerate(np.unique(ctr))}
        strat_tr    = ytr.astype(int)*10 + np.array([cohort_map[c] for c in ctr])
        dev_idx, val_idx = _tts(
            np.arange(len(ytr)), test_size=TRAIN_HOLDOUT_FRAC,
            stratify=strat_tr, random_state=cfg.RANDOM_SEED)

        log.info(f"\nTrain holdout split: dev n={len(dev_idx)}  "
                 f"(R={int(ytr[dev_idx].sum())})  |  "
                 f"val n={len(val_idx)}  (R={int(ytr[val_idx].sum())})")

        # Run pipeline on dev → val (unbiased within-training estimate)
        # seed_base offset ensures this run is statistically independent
        log.info("\n[Running pipeline: dev → val  (train eval)]")
        baseline_tr, routed_tr = _pipeline_eval(
            X_pathway_tr[dev_idx], ytr[dev_idx],
            atr[dev_idx], anames, pathway_names,
            X_pathway_tr[val_idx], ytr[val_idx], atr[val_idx],
            log=log, label='train_eval', seed_base=cfg.RANDOM_SEED + 50000,
        )

        # ── TEST EVAL: full train → held-out cohort ───────────────────────────
        # Fully independent pipeline run on the complete training set.
        log.info("\n[Running pipeline: full_train → held-out cohort  (test eval)]")
        baseline_te, routed_te = _pipeline_eval(
            X_pathway_tr, ytr,
            atr, anames, pathway_names,
            X_pathway_te, yte, ate,
            log=log, label='test_eval', seed_base=cfg.RANDOM_SEED,
        )

        # ── Log results ───────────────────────────────────────────────────────
        for cal_mode in CALIBRATION_MODES:
            log.info(f"\n  ═══ {held_out} | Calibration: {cal_mode} ═══")
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

        all_results.append({
            'held_out':      held_out,
            'n_test':        int(len(yte)),   'n_R_test':  int(yte.sum()),
            'n_dev':         int(len(dev_idx)),'n_val':     int(len(val_idx)),
            'n_R_dev':       int(ytr[dev_idx].sum()),
            'n_R_val':       int(ytr[val_idx].sum()),
            'routed_te':     routed_te,    'baseline_te': baseline_te,
            'routed_tr':     routed_tr,    'baseline_tr': baseline_tr,
            'time_seconds':  time.time()-t_fold,
        })
        if cfg.SAVE_CHECKPOINTS:
            p = cfg.OUTPUT_DIR / f'loco_pw_v5_checkpoint_{held_out}.pkl'
            with open(p, 'wb') as f: pickle.dump(all_results[-1], f)

    # ── Summary CSV ───────────────────────────────────────────────────────────
    rows = []
    for res in all_results:
        ho = res['held_out']
        for cal_mode in CALIBRATION_MODES:
            for obj in cfg.OPTUNA_OBJECTIVES:
                for mname in PIPELINE_MODELS:
                    for strat in cfg.THRESHOLD_STRATEGIES:
                        base = {'held_out':ho, 'calibration':cal_mode,
                                'objective':obj, 'model':mname,
                                'threshold_strategy':strat,
                                'n_test':res['n_test'], 'n_R_test':res['n_R_test'],
                                'n_dev':res['n_dev'],   'n_val':res['n_val']}
                        rows.append({**base, 'type':'Test_Routed',
                                     **res['routed_te'][cal_mode][obj][mname][strat]})
                        rows.append({**base, 'type':'Test_Baseline',
                                     **res['baseline_te'][cal_mode][obj][mname][strat]})
                        rows.append({**base, 'type':'Train_Routed_HoldOut',
                                     **res['routed_tr'][cal_mode][obj][mname][strat]})
                        rows.append({**base, 'type':'Train_Baseline_HoldOut',
                                     **res['baseline_tr'][cal_mode][obj][mname][strat]})

    df = pd.DataFrame(rows)
    log.info(f"\n{'='*70}")
    log.info("LOCO SUMMARY — MCC objective | MCC threshold | isotonic_on")
    log.info(f"{'='*70}")
    sub = df[(df.objective=='MCC')&(df.threshold_strategy=='MCC')&(df.calibration=='isotonic_on')]
    log.info("\n" + sub[['type','held_out','model','AUC','MCC',
                          'Sensitivity','Specificity','TP','TN','FP','FN'
                          ]].to_string(index=False))
    cfg.OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    df.to_csv(cfg.OUTPUT_DIR / 'loco_pathway_v5_metrics.csv', index=False)
    with open(cfg.OUTPUT_DIR / 'loco_pathway_v5_results.pkl', 'wb') as f:
        pickle.dump(all_results, f)
    log.info("\nSaved to output/loco_pathway_v5_metrics.csv")
    log.info(f"Total LOCO: {(time.time()-t0)/60:.1f} min")
    return all_results


if __name__ == '__main__':
    run_loco()