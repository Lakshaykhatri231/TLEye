"""Linear probe fitting for Stage 2: ridge on standardized features, folds grouped by problem, closed form so an alpha grid is cheap.
Binary targets are scored by AUROC of the ridge score; numeric targets by R^2. Probes show what is linearly decodable, not what the model uses."""
import numpy as np

ALPHAS = (1.0, 10.0, 100.0, 1000.0, 10000.0)


def group_folds(groups, k=5, seed=0):
    """Assign whole groups to k folds. No group is ever split across folds."""
    u = np.unique(groups); rng = np.random.default_rng(seed); rng.shuffle(u)
    fold_of = {g: i % k for i, g in enumerate(u)}
    return np.array([fold_of[g] for g in groups])


class Ridge:
    """Multi-alpha ridge through an SVD of the standardized training matrix."""

    def __init__(self, X, keep=None):
        """keep = number of principal components (for problem-level targets, where 896 features would overfit about 150 problems)."""
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        Z = (X - self.mu) / self.sd
        self.U, self.s, self.Vt = np.linalg.svd(Z, full_matrices=False)
        if keep is not None:
            self.U, self.s, self.Vt = self.U[:, :keep], self.s[:keep], self.Vt[:keep]

    def predict(self, y, Xq, alpha):
        ym = y.mean(); d = self.s / (self.s ** 2 + alpha)
        w = self.Vt.T @ (d * (self.U.T @ (y - ym)))
        return ((Xq - self.mu) / self.sd) @ w + ym


def auroc(y, s):
    """Rank AUROC with tie handling (no sklearn dependency here)."""
    y = np.asarray(y).astype(bool); n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort"); r = np.empty(len(s), float); ss = np.asarray(s)[order]; i = 0
    while i < len(ss):
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2 + 1; i = j + 1
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def r2(y, p):
    t = ((y - y.mean()) ** 2).sum()
    return float(1 - ((y - p) ** 2).sum() / t) if t > 0 else float("nan")


def demeaned_corr(y, p, groups):
    """Correlation after removing each problem's mean: how well a probe tracks the target WITHIN a problem (identity removed)."""
    y, p = np.asarray(y, float).copy(), np.asarray(p, float).copy()
    for g in np.unique(groups):
        m = groups == g
        y[m] -= y[m].mean(); p[m] -= p[m].mean()
    if y.std() == 0 or p.std() == 0:
        return float("nan")
    return float(np.corrcoef(y, p)[0, 1])


def score(kind, y, p, groups=None):
    return auroc(y, p) if kind == "binary" else r2(y, p)


def cv_select(X, y, groups, kind, k=5, alphas=ALPHAS, keep=None):
    """Best alpha by grouped CV: returns (alpha, cv score, out-of-fold predictions at that alpha)."""
    folds = group_folds(groups, k); oof = {a: np.zeros(len(y)) for a in alphas}
    for f in range(k):
        tr, te = folds != f, folds == f
        m = Ridge(X[tr], keep)
        for a in alphas:
            oof[a][te] = m.predict(y[tr], X[te], a)
    sc = {a: score(kind, y, oof[a], groups) for a in alphas}
    best = max(sc, key=lambda a: -np.inf if np.isnan(sc[a]) else sc[a])
    return best, sc[best], oof[best]


def shuffle_within_groups(y, groups, seed=0):
    rng = np.random.default_rng(seed); out = np.array(y, copy=True)
    for g in np.unique(groups):
        i = np.where(groups == g)[0]; out[i] = out[rng.permutation(i)]
    return out


def permute_across_groups(y, groups, seed=0):
    """For problem-level targets: give each problem another problem's values (a within-problem shuffle would change nothing)."""
    rng = np.random.default_rng(seed); u = np.unique(groups); perm = rng.permutation(u)
    val = {g: float(np.mean(y[groups == g])) for g in u}
    return np.array([val[perm[np.where(u == g)[0][0]]] for g in groups])


def holm(pvals):
    """Holm step-down adjusted p-values, in the input order."""
    p = np.asarray(pvals, float); n = len(p); order = np.argsort(p); adj = np.empty(n); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (n - rank) * p[i])); adj[i] = run
    return adj
