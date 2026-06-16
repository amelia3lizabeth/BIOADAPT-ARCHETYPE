"""
BIOADAPT-ARCHETYPE v6: LOCO Validation — Gene Level (structural v2)
====================================================================
Adds structural parity with the pathway LOCO pipeline:

  PIPELINE_MODELS    : LR, RF, SVM, EN — both baseline and routed
  CALIBRATION_MODES  : isotonic_on | isotonic_off | platt
  FIX-1              : F2/F3 min Spec >= 0.10 in find_threshold
  TRAIN EVAL         : stratified 20% of LOCO training set withheld
                       before any training begins (consensus pipeline pattern)
  CSV type labels    : Test_Routed | Test_Baseline |
                       Train_Routed_HoldOut | Train_Baseline_HoldOut

Feature selection / dimensionality reduction is UNCHANGED:
  - variance filter, axis exclusion, global MI selection
  - ArchetypeExpert: bootstrap MI selection, ps_idx, top_local
  - make_archetype_objective: pathway-space LR tuning
  - build_loco_stable_prescreen
"""
import sys, time, logging, pickle, warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split as _tts
warnings.filterwarnings('ignore')

import config as cfg
from data_loader import (load_pooled, fit_train_normalisation,
                          apply_train_normalisation, fit_variance_filter)

# Import all structural additions from updated pipeline.py
from pipeline_v2 import (
    setup_logging, get_logger,
    PIPELINE_MODELS, CALIBRATION_MODES, F2F3_MIN_SPEC, TRAIN_HOLDOUT_FRAC,
    find_threshold, compute_metrics,
    fit_calibrator, apply_calibrator,
    _gene_pipeline_eval,
)


# ─── LOCO STABLE PRESCREEN (UNCHANGED) ───────────────────────────────────────
def build_loco_stable_prescreen(Xtr, ctr, genes):
    """
    Returns pathway gene indices surviving the variance filter on the
    LOCO training set (all cohorts except held-out).
    """
    all_pathway_genes = set(
        g for gs in cfg.AXIS_DEFINITIONS.values() for g in gs)
    vm  = fit_variance_filter(Xtr, percentile=cfg.VAR_FILTER_PCT, cohort_train=ctr)
    gv  = [g for g, m in zip(genes, vm) if m]
    idx = sorted(i for i, g in enumerate(gv) if g in all_pathway_genes)
    return np.array(idx, dtype=int), gv, vm


# ─── LOCO PIPELINE ───────────────────────────────────────────────────────────
def run_loco():
    log = setup_logging('loco_gene_v2')
    t0  = time.time()
    axis_genes = sorted(set(g for gs in cfg.AXIS_DEFINITIONS.values() for g in gs))
    axis_set   = set(axis_genes)

    log.info("="*62)
    log.info(" BIOADAPT-ARCHETYPE v6 — LOCO GENE PIPELINE (structural v2)")
    log.info(f" Cohorts       : {cfg.LOCO_COHORTS}")
    log.info(f" Models        : {PIPELINE_MODELS}")
    log.info(f" Calibration   : {CALIBRATION_MODES}")
    log.info(f" FIX-1         : F2/F3 min Spec >= {F2F3_MIN_SPEC}")
    log.info(f" Train holdout : {int(TRAIN_HOLDOUT_FRAC*100)}% of LOCO train set")
    log.info(f" Feature sel   : variance filter + MI (UNCHANGED)")
    log.info("="*62)

    X, y, cohort, genes, _ = load_pooled(cfg.LOCO_COHORTS, common_genes_only=True)
    log.info(f"Loaded {X.shape[0]} x {X.shape[1]}")
    for cid in np.unique(cohort):
        m = cohort==cid
        log.info(f"  {cid}: n={m.sum()}  R={int(y[m].sum())}  resp={y[m].mean():.2f}")

    all_results = []

    for held_out in cfg.LOCO_COHORTS:
        t_fold = time.time()
        log.info(f"\n{'─'*62}\nHELD-OUT: {held_out}\n{'─'*62}")

        mho = cohort==held_out; mtr = ~mho
        Xtr, Xte = X[mtr], X[mho]
        ytr, yte = y[mtr], y[mho]
        ctr, cte = cohort[mtr], cohort[mho]

        log.info(f"Train: {dict(zip(*np.unique(ctr, return_counts=True)))}")
        log.info(f"Test:  n={len(yte)}  R={int(yte.sum())}  NR={int((yte==0).sum())}")
        if yte.sum()==0 or (1-yte).sum()==0:
            log.warning("Skipping: single class in test"); continue

        # ── Stable prescreen on full LOCO training set ────────────────────────
        # Used only for the test eval call. Train eval uses per-dev prescreen.
        pathway_ps_idx, gv_full, vm_full = build_loco_stable_prescreen(Xtr, ctr, genes)
        stable_genes_for_test = [gv_full[i] for i in pathway_ps_idx]
        log.info(f"LOCO stable prescreen: {len(stable_genes_for_test)} pathway genes")

        # ── Train holdout split — reserved BEFORE any model training ─────────
        # Stratified by cohort × response so all training cohorts are
        # represented in both dev and val.
        cohort_map  = {c: i for i, c in enumerate(np.unique(ctr))}
        strat_tr    = ytr.astype(int)*10 + np.array([cohort_map[c] for c in ctr])
        dev_idx, val_idx = _tts(
            np.arange(len(ytr)), test_size=TRAIN_HOLDOUT_FRAC,
            stratify=strat_tr, random_state=cfg.RANDOM_SEED)

        log.info(f"Train holdout: dev n={len(dev_idx)} (R={int(ytr[dev_idx].sum())}) | "
                 f"val n={len(val_idx)} (R={int(ytr[val_idx].sum())})")

        # ── Pipeline call 1: dev → val (unbiased within-training eval) ───────
        log.info("\n[Gene pipeline: dev → val  (train eval)]")
        baseline_tr, routed_tr, gene_data_tr = _gene_pipeline_eval(
            Xtr[dev_idx], ytr[dev_idx], ctr[dev_idx],
            Xtr[val_idx], ytr[val_idx], ctr[val_idx],
            genes, axis_set,
            stable_genes=None,   # per-dev prescreen — no stable list for train eval
            log=log, label=f'train_eval_{held_out}',
            seed_base=cfg.RANDOM_SEED + 50000,
        )

        # ── Pipeline call 2: full LOCO train → held-out cohort ───────────────
        log.info("\n[Gene pipeline: full_train → held-out cohort  (test eval)]")
        baseline_te, routed_te, gene_data_te = _gene_pipeline_eval(
            Xtr, ytr, ctr,
            Xte, yte, cte,
            genes, axis_set,
            stable_genes=stable_genes_for_test,
            log=log, label=f'test_eval_{held_out}',
            seed_base=cfg.RANDOM_SEED,
        )

        # ── Log results ───────────────────────────────────────────────────────
        log.info(f"\n  Results on {held_out}:")
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
            'gene_data_te':  gene_data_te, # experts on full LOCO train — use for discovery
            'gene_data_tr':  gene_data_tr, # experts on dev split only
            'stable_genes_n':len(stable_genes_for_test),
            'time_seconds':  time.time()-t_fold,
        })
        if cfg.SAVE_CHECKPOINTS:
            p = cfg.OUTPUT_DIR / f'loco_gene_v2_checkpoint_{held_out}.pkl'
            with open(p, 'wb') as f: pickle.dump(all_results[-1], f)
            log.info(f"Checkpoint saved → {p}")

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
    log.info("LOCO GENE SUMMARY — MCC objective | MCC threshold | isotonic_on")
    log.info(f"{'='*70}")
    sub = df[(df.objective=='MCC')&(df.threshold_strategy=='MCC')&(df.calibration=='isotonic_on')]
    log.info("\n" + sub[['type','held_out','model','AUC','MCC',
                          'Sensitivity','Specificity','TP','TN','FP','FN'
                          ]].to_string(index=False))
    # ── Discovery gene aggregation (from test-eval experts — full LOCO train) ──
    log.info(f"\n{'─'*62}\nDISCOVERY GENES (aggregated across {len(all_results)} LOCO folds)\n{'─'*62}")
    from collections import Counter as _Counter
    disc = defaultdict(_Counter)
    wf   = defaultdict(list)
    for res in all_results:
        for am, gd in res.get('gene_data_te', {}).items():
            for g, f in gd['stable']:
                disc[am][g] += 1
            wf[am].append(gd['frequencies'])

    disc_rows = []
    for am, ctr_d in sorted(disc.items()):
        top_str = [f"{g}({n}/{len(all_results)})" for g, n in ctr_d.most_common(10)]
        log.info(f"  {am}: {top_str}")
        for g, fc in ctr_d.most_common():
            bfreqs = [fd.get(g, 0) / cfg.N_BOOTSTRAP for fd in wf[am] if g in fd]
            disc_rows.append({
                'archetype_model':     am,
                'gene':                g,
                'loco_fold_frequency': fc,
                'n_loco_folds':        len(all_results),
                'mean_bootstrap_freq': float(np.mean(bfreqs)) if bfreqs else 0.0,
            })

    cfg.OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    df.to_csv(cfg.OUTPUT_DIR / 'loco_gene_v2_metrics.csv', index=False)
    pd.DataFrame(disc_rows).to_csv(cfg.OUTPUT_DIR / 'loco_gene_v2_discovery.csv', index=False)
    with open(cfg.OUTPUT_DIR / 'loco_gene_v2_results.pkl', 'wb') as f:
        pickle.dump(all_results, f)
    log.info("\nSaved to output/loco_gene_v2_metrics.csv")
    log.info("Saved to output/loco_gene_v2_discovery.csv")
    log.info(f"Total LOCO: {(time.time()-t0)/60:.1f} min")
    return all_results


if __name__ == '__main__':
    run_loco()