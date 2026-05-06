"""
Transform-agnostic distinguishers between real-Q and Q-sub feature
distributions. The null is "feature distribution under (G, Q=k*G) is
identical to under (G, R=r*G with r != k)".

Two tests:

1. MMD (maximum mean discrepancy) with RBF kernel + permutation test for
   p-value. Standard biased estimator. Reports raw MMD^2 and permutation p.

2. Held-out classifier AUC. Train logistic regression on standardized
   features to discriminate real-Q from qsub. Report mean cross-validated
   ROC-AUC and a permutation p-value via label shuffling.

Inputs are arrays X_real (n x d) and X_qsub (n x d) of features. We pair
each real-Q trial with its Q-sub counterpart in the experiment so that
n_real == n_qsub, but the tests treat them as two unpaired samples.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.preprocessing import StandardScaler


def _rbf_kernel(X: np.ndarray, Y: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian RBF kernel."""
    XX = (X * X).sum(1)[:, None]
    YY = (Y * Y).sum(1)[None, :]
    sqdist = XX + YY - 2 * X @ Y.T
    sqdist = np.clip(sqdist, 0, None)
    return np.exp(-sqdist / (2 * sigma ** 2))


def _median_heuristic(X: np.ndarray, Y: np.ndarray) -> float:
    Z = np.vstack([X, Y])
    n = len(Z)
    if n > 200:
        idx = np.random.default_rng(0).choice(n, 200, replace=False)
        Z = Z[idx]
    diffs = Z[:, None, :] - Z[None, :, :]
    sqdist = (diffs ** 2).sum(-1)
    iu = np.triu_indices(len(Z), k=1)
    med = float(np.sqrt(np.median(sqdist[iu])))
    return max(med, 1e-6)


def mmd2_with_pvalue(X_real: np.ndarray, X_qsub: np.ndarray, *,
                      n_permutations: int = 200, seed: int = 0) -> dict:
    """Biased MMD^2 estimator + permutation p-value."""
    if X_real.shape != X_qsub.shape:
        # Truncate to common
        n = min(len(X_real), len(X_qsub))
        X_real = X_real[:n]
        X_qsub = X_qsub[:n]
    if len(X_real) < 4:
        return dict(mmd2=0.0, pval=1.0, sigma=0.0)
    sigma = _median_heuristic(X_real, X_qsub)
    Kxx = _rbf_kernel(X_real, X_real, sigma)
    Kyy = _rbf_kernel(X_qsub, X_qsub, sigma)
    Kxy = _rbf_kernel(X_real, X_qsub, sigma)
    mmd2 = float(Kxx.mean() + Kyy.mean() - 2 * Kxy.mean())
    # Permutation test
    Z = np.vstack([X_real, X_qsub])
    n = len(X_real)
    rng = np.random.default_rng(seed)
    null = np.empty(n_permutations)
    K_full = _rbf_kernel(Z, Z, sigma)
    for i in range(n_permutations):
        perm = rng.permutation(2 * n)
        a, b = perm[:n], perm[n:]
        Kxx_p = K_full[np.ix_(a, a)].mean()
        Kyy_p = K_full[np.ix_(b, b)].mean()
        Kxy_p = K_full[np.ix_(a, b)].mean()
        null[i] = Kxx_p + Kyy_p - 2 * Kxy_p
    pval = float((null >= mmd2).mean())
    return dict(mmd2=mmd2, pval=pval, sigma=sigma)


def classifier_auc(X_real: np.ndarray, X_qsub: np.ndarray, *,
                   n_permutations: int = 100, seed: int = 0,
                   n_splits: int = 5) -> dict:
    """Group-aware K-fold logistic regression AUC + label-permutation p-value.

    Critical: when X_real and X_qsub are pairwise per-instance (i-th row of
    each comes from instance i), a vanilla StratifiedKFold leaks paired
    instances across train and test, which biases AUC toward 0 when features
    are similar within pairs. We use GroupKFold with group=instance_index to
    keep both members of each pair in the same fold.
    """
    if X_real.shape != X_qsub.shape:
        n = min(len(X_real), len(X_qsub))
        X_real = X_real[:n]
        X_qsub = X_qsub[:n]
    n = len(X_real)
    if n < 8:
        return dict(auc=0.5, pval=1.0)
    X = np.vstack([X_real, X_qsub])
    y = np.concatenate([np.ones(n), np.zeros(n)])
    groups = np.concatenate([np.arange(n), np.arange(n)])  # pair index
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    var = X.var(axis=0)
    X = X[:, var > 1e-12]
    if X.shape[1] == 0:
        return dict(auc=0.5, pval=1.0)

    n_groups = n
    n_splits_eff = min(n_splits, n_groups)
    if n_splits_eff < 2:
        return dict(auc=0.5, pval=1.0)

    def cv_auc(X_, y_, groups_, rs):
        # GroupKFold has no random_state; for permutation seed effect we
        # shuffle the *group ids* deterministically by rs.
        rng_local = np.random.default_rng(rs)
        perm = rng_local.permutation(np.arange(n_groups))
        # remap group ids by perm so different rs => different fold assignment
        groups_remapped = perm[groups_]
        gkf = GroupKFold(n_splits=n_splits_eff)
        aucs = []
        for train, test in gkf.split(X_, y_, groups=groups_remapped):
            if len(set(y_[test])) < 2:
                continue
            try:
                clf = LogisticRegression(max_iter=1000, C=0.1)
                clf.fit(X_[train], y_[train])
                p = clf.predict_proba(X_[test])[:, 1]
                aucs.append(roc_auc_score(y_[test], p))
            except Exception:
                aucs.append(0.5)
        return float(np.mean(aucs)) if aucs else 0.5

    auc = cv_auc(X, y, groups, seed)
    rng = np.random.default_rng(seed)
    null = np.empty(n_permutations)
    for i in range(n_permutations):
        y_perm = rng.permutation(y)
        null[i] = cv_auc(X, y_perm, groups, seed + i + 1)
    pval = float((null >= auc).mean())
    return dict(auc=auc, pval=pval)
