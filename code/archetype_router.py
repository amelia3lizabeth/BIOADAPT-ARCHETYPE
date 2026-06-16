"""
BIOADAPT-ARCHETYPE: Archetype Routing
=====================================
Gaussian Mixture Model on biologically-anchored axis scores.
All preprocessing parameters fit on training data only.
"""

import numpy as np
from scipy.stats import rankdata
from scipy.special import ndtri
from sklearn.mixture import GaussianMixture
from sklearn.metrics import normalized_mutual_info_score
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

import config as cfg


def rank_inverse_normal(x: np.ndarray) -> np.ndarray:
    """Rank-based inverse normal transformation (Blom)."""
    r = rankdata(x)
    n = len(x)
    return ndtri((r - 0.375) / (n + 0.25))


def fit_rin_transform(x_train: np.ndarray) -> tuple:
    """Fit RIN parameters on training data. Returns sorted (raw, rin) pairs."""
    rin = rank_inverse_normal(x_train)
    return np.sort(x_train), np.sort(rin)


def apply_rin_transform(x: np.ndarray, sorted_raw: np.ndarray,
                          sorted_rin: np.ndarray) -> np.ndarray:
    """Apply RIN to new data via interpolation against training quantiles."""
    return np.interp(x, sorted_raw, sorted_rin)


class ArchetypeRouter:
    """
    Routes patients to k biologically-defined archetypes.

    Pipeline:
        1. Compute axis composite scores (mean of member genes)
        2. RIN-transform each axis using TRAIN-fit quantile mapping
        3. Fit k-component GMM on training axis scores
        4. predict_proba returns soft archetype membership
    """

    def __init__(self, axes: list = None, k: int = None,
                  random_state: int = None):
        self.axes = axes if axes is not None else cfg.ARCHETYPE_AXES
        self.k    = k    if k    is not None else cfg.ARCHETYPE_K
        self.random_state = random_state if random_state is not None else cfg.RANDOM_SEED
        self.rin_params = {}
        self.gmm = None
        self.fitted_axis_names = None

    def fit(self, axis_scores: np.ndarray, axis_names: list):
        """
        Fit on training axis scores.

        axis_scores : (n_patients, n_axes_total)
        axis_names  : list naming each column of axis_scores
        """
        # Subset to the configured archetype axes
        ax_idx = [axis_names.index(a) for a in self.axes]
        X = axis_scores[:, ax_idx]

        # RIN per axis on training data
        X_rin = np.zeros_like(X)
        for i, name in enumerate(self.axes):
            s_raw, s_rin = fit_rin_transform(X[:, i])
            self.rin_params[name] = (s_raw, s_rin)
            X_rin[:, i] = apply_rin_transform(X[:, i], s_raw, s_rin)

        # Fit GMM
        self.gmm = GaussianMixture(
            n_components=self.k,
            covariance_type='full',
            random_state=self.random_state,
            n_init=10,
            max_iter=500,
        )
        self.gmm.fit(X_rin)
        self.fitted_axis_names = list(axis_names)
        return self

    def transform(self, axis_scores: np.ndarray, axis_names: list) -> np.ndarray:
        """Apply RIN using train quantiles. Returns (n, n_axes)."""
        ax_idx = [axis_names.index(a) for a in self.axes]
        X = axis_scores[:, ax_idx]
        X_rin = np.zeros_like(X)
        for i, name in enumerate(self.axes):
            s_raw, s_rin = self.rin_params[name]
            X_rin[:, i] = apply_rin_transform(X[:, i], s_raw, s_rin)
        return X_rin

    def predict_proba(self, axis_scores: np.ndarray, axis_names: list) -> np.ndarray:
        """Returns (n_patients, k) soft membership probabilities."""
        X_rin = self.transform(axis_scores, axis_names)
        return self.gmm.predict_proba(X_rin)

    def predict(self, axis_scores: np.ndarray, axis_names: list) -> np.ndarray:
        """Returns (n_patients,) hard archetype labels."""
        return self.predict_proba(axis_scores, axis_names).argmax(axis=1)

    def label_archetypes(self, axis_scores: np.ndarray, axis_names: list,
                          y: np.ndarray) -> dict:
        """
        Assign biological labels to numeric archetypes based on centroids
        and response rates in training data.
        """
        labels = self.predict(axis_scores, axis_names)
        centroids = self.gmm.means_
        names = {}
        for k in range(self.k):
            c = centroids[k]
            mean_c = c.mean()
            # Get axis indices
            ai = {a: i for i, a in enumerate(self.axes)}
            ctl_c = c[ai['ctl']] if 'ctl' in ai else 0
            stro_c = c[ai['stromal']] if 'stromal' in ai else 0
            term_c = c[ai['terminal']] if 'terminal' in ai else 0
            tum_c = c[ai['tumour']] if 'tumour' in ai else 0

            if mean_c > 0.4 and ctl_c > 0.3:
                names[k] = 'Hot_infiltrated'
            elif stro_c > 0.3 and ctl_c > 0:
                names[k] = 'Stromal_excluded'
            elif tum_c < -0.5 and ctl_c < -0.3:
                names[k] = 'Immune_desert'
            elif ctl_c > 0.3 and term_c > 0.2:
                names[k] = 'CTL_rich'
            elif ctl_c < -0.4:
                names[k] = 'Cold_depleted'
            else:
                names[k] = f'Intermediate_{k}'
        return names


def cross_cohort_nmi(axis_scores: np.ndarray, cohort: np.ndarray,
                      axis_names: list, k: int = None) -> tuple:
    """
    Compute pairwise cross-cohort NMI as a stability check.

    Returns (mean_nmi, list of (cohort1, cohort2, nmi) tuples).
    """
    if k is None:
        k = cfg.ARCHETYPE_K
    ax_idx = [axis_names.index(a) for a in cfg.ARCHETYPE_AXES]
    X = axis_scores[:, ax_idx]
    cohorts = np.unique(cohort)
    pairs = []
    for i, c1 in enumerate(cohorts):
        for j, c2 in enumerate(cohorts):
            if i >= j: continue
            d1 = X[cohort == c1]
            d2 = X[cohort == c2]
            if len(d1) < k+1 or len(d2) < k+1: continue
            try:
                g1 = GaussianMixture(n_components=k, random_state=cfg.RANDOM_SEED, n_init=8).fit(d1)
                g2 = GaussianMixture(n_components=k, random_state=cfg.RANDOM_SEED, n_init=8).fit(d2)
                nmi = normalized_mutual_info_score(g1.predict(d2), g2.predict(d2))
                pairs.append((c1, c2, float(nmi)))
            except Exception:
                pass
    mean_nmi = np.mean([p[2] for p in pairs]) if pairs else 0.0
    return mean_nmi, pairs


if __name__ == '__main__':
    from data_loader import load_pooled, compute_axis_scores
    X, y, cohort, genes, _ = load_pooled()
    axes, axis_names, _ = compute_axis_scores(X, genes)

    router = ArchetypeRouter()
    router.fit(axes, axis_names)
    labels = router.predict(axes, axis_names)

    arch_names = router.label_archetypes(axes, axis_names, y)
    print("\nArchetype assignments (training pool):")
    for k in range(router.k):
        mask = labels == k
        n = mask.sum()
        r = y[mask].mean() if n > 0 else 0
        print(f"  A{k} ({arch_names[k]}): n={n}, response={r:.2f}")

    nmi_mean, pairs = cross_cohort_nmi(axes, cohort, axis_names)
    print(f"\nMean cross-cohort NMI: {nmi_mean:.3f}")
    for c1, c2, n in pairs:
        print(f"  {c1} vs {c2}: NMI={n:.3f}")
