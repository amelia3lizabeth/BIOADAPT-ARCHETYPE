"""
BIOADAPT-ARCHETYPE v6: Smoke Test
===================================
Verifies all imports, data loading, pathway prescreen construction,
and a single-bootstrap expert fit/predict cycle before committing to
a full 5-fold run.
"""
import sys
sys.path.insert(0, '.')
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("── Import check ─────────────────────────────────────────")
import config as cfg
print(f"  PRESCREEN_MODE        = {cfg.PRESCREEN_MODE}")
print(f"  USE_ARCHETYPE_PCA     = {cfg.USE_ARCHETYPE_PCA}")
print(f"  MODELS                = {cfg.MODELS}")
print(f"  N_FEATURES_PER_ARCH   = {cfg.N_FEATURES_PER_ARCHETYPE}")
print(f"  ARCHETYPE_K           = {cfg.ARCHETYPE_K}")

from data_loader import load_pooled, compute_axis_scores, fit_train_normalisation, apply_train_normalisation, fit_variance_filter
from archetype_router import ArchetypeRouter
from pipeline import (ArchetypeExpert, build_pathway_prescreen,
                       smote_data, build_classifier, find_threshold)
print("  All imports OK")

print("\n── Data loading ─────────────────────────────────────────")
X, y, cohort, genes, norm = load_pooled()
print(f"  Shape: {X.shape}  R={int(y.sum())}/{len(y)}")
print(f"  HSFX1 present: {'hsfx1' in genes}")

print("\n── Pathway prescreen ────────────────────────────────────")
all_pw = sorted(set(g for gs in cfg.AXIS_DEFINITIONS.values() for g in gs))
print(f"  Total unique pathway genes : {len(all_pw)}")
# Simulate variance filter output
np.random.seed(42)
vm = np.ones(len(genes), dtype=bool)   # all pass for smoke test
gv = list(genes)
ps_idx = build_pathway_prescreen(gv)
print(f"  Pathway genes in dataset   : {len(ps_idx)}")
pw_genes = [gv[i] for i in ps_idx]
print(f"  Sample: {pw_genes[:8]}")
expected_hit = cfg.N_FEATURES_PER_ARCHETYPE / len(ps_idx) if ps_idx.size else 0
print(f"  Expected bootstrap hit rate: {expected_hit:.1%}")

print("\n── Router smoke test ────────────────────────────────────")
np_ = fit_train_normalisation(X, cohort)
Xz  = apply_train_normalisation(X, cohort, np_)
atr, anames, npres = compute_axis_scores(Xz, genes)
print(f"  Axis scores shape: {atr.shape}")
router = ArchetypeRouter()
router.fit(atr, anames)
aprob = router.predict_proba(atr, anames)
print(f"  Archetype proba shape: {aprob.shape}")
labels = router.label_archetypes(atr, anames, y)
for k in range(router.k):
    m = aprob.argmax(1) == k
    print(f"  A{k} ({labels[k]}): n={int(m.sum())}  "
          f"resp={y[m].mean():.2f}")

print("\n── Expert smoke test (2 bootstraps) ─────────────────────")
orig_n = cfg.N_BOOTSTRAP
cfg.N_BOOTSTRAP = 2
vm2 = fit_variance_filter(X[:200], percentile=cfg.VAR_FILTER_PCT, cohort_train=cohort[:200])
Xvf = Xz[:, vm2]
gv2 = [g for g, keep in zip(genes, vm2) if keep]
ps2 = build_pathway_prescreen(gv2)
print(f"  Pathway genes after variance filter: {len(ps2)}")
mem = aprob[:, 0]
exp = ArchetypeExpert('LR', model_id=0)
params = {'C': 0.1}
exp.fit(Xz, y, mem, list(genes), params, ps_idx=ps2, fold_id=0, arch_id=0)
print(f"  Bootstraps fitted: {len(exp.bootstrap_models)}")
prob = exp.predict_proba_raw(Xz, ps_idx=ps2)
print(f"  Predict shape: {prob.shape}  range [{prob.min():.3f}, {prob.max():.3f}]")
stable = exp.stable_features(thr=0.0)
print(f"  Top genes: {[g for g,_ in stable[:5]]}")
cfg.N_BOOTSTRAP = orig_n

print("\n── SMOKE TEST PASSED ✓ ──────────────────────────────────")
print("Ready to run: python pipeline.py")
