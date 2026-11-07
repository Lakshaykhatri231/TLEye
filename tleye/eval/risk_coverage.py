"""Risk-coverage: how the error rate falls when the model may abstain on its least confident cases. Descriptive; no thresholds tuned.
    python -I -m tleye.eval.risk_coverage   -> reports/risk_coverage.md, reports/risk_coverage.svg

Rule: predict TLE when P(TLE) >= 0.5 (calibrated), confidence = max(p, 1 - p); keep the most confident share `coverage` of the test rows.
AURC = mean selective error over all coverages (lower is better). CIs: problem-clustered bootstrap."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from tleye.eval.metrics import cluster_bootstrap

COVERAGES = (1.0, 0.8, 0.6, 0.4, 0.2)
N_BOOT = 300
OUT_MD, OUT_SVG = Path("reports/risk_coverage.md"), Path("reports/risk_coverage.svg")
LORA = {"LoRA seed 0": "reports/kaggle_confirm/cfgN/test_logits.npz", "LoRA seed 1": "reports/kaggle_seeds/cfg_s1/test_logits.npz",
        "LoRA seed 2": "reports/kaggle_seeds/cfg_s2/test_logits.npz"}


def curve(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Selective error after keeping the k most confident rows, for k = 1..n."""
    order = np.argsort(-np.maximum(p, 1 - p), kind="stable")
    wrong = ((p[order] >= 0.5) != (y[order] == 1)).astype(float)
    return np.cumsum(wrong) / np.arange(1, len(y) + 1)


def aurc(y, p) -> float:
    return float(curve(y, p).mean())


def error_at(y, p, cov: float) -> float:
    c = curve(y, p)
    return float(c[max(1, int(round(cov * len(y)))) - 1])


def load() -> pd.DataFrame:
    meta = pd.read_parquet("data/processed/features_clean.parquet", columns=["submission_id", "problem_id", "y_tle"])
    d = meta.merge(pd.read_parquet("data/processed/b2m_test_preds.parquet").rename(columns={"p_raw": "B2m"})[["submission_id", "B2m"]], on="submission_id")
    for name, f in LORA.items():
        z = np.load(f)
        d = d.merge(pd.DataFrame({"submission_id": z["submission_id"].astype(str), name: 1 / (1 + np.exp(-z["logit"] / float(z["T"])))}), on="submission_id")
    return d


def svg(series: dict[str, np.ndarray], path: Path) -> None:
    """Static line chart. Palette slots 1-2 of the reference palette (validated light; dark steps via prefers-color-scheme)."""
    W, H, L, R, T, B = 640, 360, 56, 112, 40, 44
    ymax = max(float(c[int(0.05 * len(c)):].max()) for c in series.values()) * 1.1
    ymax = float(np.ceil(ymax * 20) / 20)
    def X(cov): return L + cov * (W - L - R)
    def Y(err): return T + (1 - err / ymax) * (H - T - B)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="system-ui, -apple-system, sans-serif" font-size="12">',
             "<title>Risk-coverage on the test split</title>",
             "<style>.bg{fill:#fcfcfb}.t1{fill:#0b0b0b}.t2{fill:#52514e}.mut{fill:#898781}.grid{stroke:#e1e0d9}.ax{stroke:#898781}"
             ".s1{stroke:#2a78d6}.s2{stroke:#eb6834}.f1{fill:#2a78d6}.f2{fill:#eb6834}"
             "@media (prefers-color-scheme: dark){.bg{fill:#1a1a19}.t1{fill:#f0efec}.t2{fill:#c3c2b7}.grid{stroke:#2c2c2a}.s1{stroke:#3987e5}.s2{stroke:#d95926}.f1{fill:#3987e5}.f2{fill:#d95926}}</style>",
             f'<rect class="bg" width="{W}" height="{H}"/>',
             f'<text class="t1" x="{L}" y="22" font-size="14" font-weight="600">Error rate when the model may skip its least confident cases (test split)</text>']
    for k in range(0, int(round(ymax * 20)) + 1):
        e = k / 20
        parts.append(f'<line class="grid" x1="{L}" x2="{W - R}" y1="{Y(e):.1f}" y2="{Y(e):.1f}" stroke-width="1"/>')
        parts.append(f'<text class="mut" x="{L - 8}" y="{Y(e) + 4:.1f}" text-anchor="end">{e:.0%}</text>')
    for cov in (0.2, 0.4, 0.6, 0.8, 1.0):
        parts.append(f'<text class="mut" x="{X(cov):.1f}" y="{H - B + 18}" text-anchor="middle">{cov:.0%}</text>')
    parts.append(f'<line class="ax" x1="{L}" x2="{W - R}" y1="{H - B}" y2="{H - B}" stroke-width="1"/>')
    parts.append(f'<text class="t2" x="{(L + W - R) / 2:.0f}" y="{H - 8}" text-anchor="middle">share of solutions the model answers (coverage)</text>')
    for i, (name, c) in enumerate(series.items(), 1):
        n = len(c); idx = np.unique(np.linspace(int(0.05 * n), n - 1, 200).astype(int))
        pts = " ".join(f"{X((j + 1) / n):.1f},{Y(c[j]):.1f}" for j in idx)
        parts.append(f'<polyline class="s{i}" fill="none" stroke-width="2" stroke-linejoin="round" points="{pts}"/>')
        parts.append(f'<circle class="f{i}" cx="{X(1.0):.1f}" cy="{Y(c[-1]):.1f}" r="4"/>')
        parts.append(f'<text class="t1" x="{X(1.0) + 10:.1f}" y="{Y(c[-1]) + (4 if i == 1 else 4):.1f}">{name}</text>')
    lx = L + 8
    for i, name in enumerate(series, 1):
        y0 = T + 10 + 18 * (i - 1)
        parts.append(f'<line class="s{i}" x1="{lx}" x2="{lx + 18}" y1="{y0}" y2="{y0}" stroke-width="2"/><text class="t2" x="{lx + 24}" y="{y0 + 4}">{name}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n")


def main() -> None:
    d = load()
    y, g = d.y_tle.values, d.problem_id.values
    models = ["B2m", *LORA]
    fm = lambda v, c: f"{v:.3f} [{c[0]:.3f}, {c[1]:.3f}]"  # noqa: E731
    L = ["# Risk-coverage on the test split (script-generated; descriptive, nothing tuned)", "",
         f"{len(d):,} test submissions, {d.problem_id.nunique()} problems. Predict TLE when calibrated P(TLE) >= 0.5; the model answers only its most confident "
         "share of cases (coverage) and skips the rest. Selective error = error rate on the answered cases. AURC = mean selective error over all coverages "
         f"(lower is better). 95% CIs: problem-clustered bootstrap ({N_BOOT}).", "",
         "| model | AURC | " + " | ".join(f"error at {c:.0%} coverage" for c in COVERAGES) + " |", "|---|---|" + "---|" * len(COVERAGES)]
    for m in models:
        p = d[m].values
        ci = cluster_bootstrap(g, lambda i, p=p: aurc(y[i], p[i]), n_boot=N_BOOT)
        L.append(f"| {m} | {fm(aurc(y, p), ci)} | " + " | ".join(f"{error_at(y, p, c):.3f}" for c in COVERAGES) + " |")
    a, b = d["LoRA seed 0"].values, d["B2m"].values
    lo, hi = cluster_bootstrap(g, lambda i: aurc(y[i], a[i]) - aurc(y[i], b[i]), n_boot=N_BOOT)
    L += ["", f"Paired AURC difference, LoRA seed 0 minus B2m: {aurc(y, a) - aurc(y, b):+.3f} [{lo:+.3f}, {hi:+.3f}] (negative = LoRA makes fewer errors).", "",
          "Reading it: at 100% coverage the error is the plain error rate at P(TLE) >= 0.5. Abstaining lowers it, which is what SUSPECTED TLE and an "
          "uncertain LIKELY AC are for. The probabilities are calibrated to a sample with about 40% TLE, so the 0.5 rule is not tuned for real-world rates.", "",
          "![risk-coverage](risk_coverage.svg)"]
    svg({"LoRA (seed 0)": curve(y, a), "LightGBM B2m": curve(y, b)}, OUT_SVG)
    OUT_MD.write_text("\n".join(L) + "\n"); print("\n".join(L))


if __name__ == "__main__":
    main()
