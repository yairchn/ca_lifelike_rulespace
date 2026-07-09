"""Build the (Δ, Δ_edge) histogram figure: Σ>0 vs Σ=0 distributions, two panels.

Δ takes 19 discrete values (multiples of 1/9 in [-1, 1]); Δ_edge takes 17
(multiples of 1/8). Both populations are plotted as bar charts at the discrete
support values, normalised to density so the rare Σ>0 cohort can be visually
compared to the Σ=0 bulk.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
COORDS = ROOT / "data" / "symmetry_coords_full.parquet"
OUT = ROOT / "manuscript" / "figures" / "Fig_delta_histograms.png"


def discrete_hist(values, support, weights=None):
    """Density per support value. values can be repeated; support is the discrete
    grid; returns p[i] = N(values == support[i]) / N_total."""
    vals = np.asarray(values, dtype=np.float64)
    p = np.zeros(len(support), dtype=np.float64)
    for i, s in enumerate(support):
        p[i] = np.isclose(vals, s, atol=1e-6).sum()
    p /= max(1.0, p.sum())
    return p


def main():
    df = pd.read_parquet(COORDS)
    print(f"loaded {len(df)} rules, Σ>0 = {df['sig'].sum()}")
    pos = df[df["sig"]]; neg = df[~df["sig"]]

    delta_support = np.arange(-9, 10) / 9.0     # 19 values
    edge_support  = np.arange(-8,  9) / 8.0     # 17 values

    pos_delta = discrete_hist(pos["delta"].values, delta_support)
    neg_delta = discrete_hist(neg["delta"].values, delta_support)
    pos_edge  = discrete_hist(pos["delta_edge"].values, edge_support)
    neg_edge  = discrete_hist(neg["delta_edge"].values, edge_support)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))

    # --- Panel a: Δ ---
    ax = axes[0]
    w = 1.0 / 9.0 * 0.42   # width relative to spacing
    ax.bar(delta_support - w/2, neg_delta, width=w, color="#7a7a7a",
           alpha=0.55, edgecolor="black", linewidth=0.3,
           label=f"Σ=0  (n={len(neg)})")
    ax.bar(delta_support + w/2, pos_delta, width=w, color="crimson",
           alpha=0.85, edgecolor="black", linewidth=0.3,
           label=f"Σ>0  (n={len(pos)})")
    ax.axvline(0, color="black", linestyle="--", linewidth=0.6, alpha=0.5)
    pos_med = float(np.median(pos["delta"]))
    ax.axvline(pos_med, color="crimson", linestyle=":", linewidth=1.2,
               label=f"Σ>0 median = {pos_med:+.3f}")
    ax.set_xlabel(r"$\delta = (|B| - |S|)/9$")
    ax.set_ylabel("density")
    ax.set_title(r"Cardinality imbalance $\delta$"
                 "\n"
                 r"$\Sigma>0$ skewed to $\delta<0$ (340/424 below zero)")
    ax.legend(fontsize=8.5, loc="upper left")
    ax.set_xlim(-1.05, 1.05)

    # --- Panel b: Δ_edge ---
    ax = axes[1]
    w = 1.0 / 8.0 * 0.42
    ax.bar(edge_support - w/2, neg_edge, width=w, color="#7a7a7a",
           alpha=0.55, edgecolor="black", linewidth=0.3,
           label=f"Σ=0  (n={len(neg)})")
    ax.bar(edge_support + w/2, pos_edge, width=w, color="crimson",
           alpha=0.85, edgecolor="black", linewidth=0.3,
           label=f"Σ>0  (n={len(pos)})")
    ax.axvline(0, color="black", linestyle="--", linewidth=0.6, alpha=0.5)
    pos_med_e = float(np.median(pos["delta_edge"]))
    ax.axvline(pos_med_e, color="crimson", linestyle=":", linewidth=1.2,
               label=f"Σ>0 median = {pos_med_e:+.3f}")
    # Rank-biserial effect size (Cliff's delta) — descriptive, no p-value
    # on a fully enumerated population.  Sign convention: r = 2*P - 1
    # where P = P(Sigma>0 sample exceeds Sigma=0 sample).  Negative r means
    # Sigma>0 shifted below Sigma=0.
    from scipy.stats import mannwhitneyu
    U, _ = mannwhitneyu(pos["delta_edge"].values, neg["delta_edge"].values,
                        alternative="two-sided")
    n1, n2 = len(pos), len(neg)
    r_rb = 2.0 * U / (n1 * n2) - 1.0
    ax.set_xlabel(r"$\delta_{\mathrm{edge}} = (e(B) - e(S))/8$")
    ax.set_ylabel("density")
    ax.set_title(r"Edge imbalance $\delta_{\mathrm{edge}}$ "
                 r"($\tau$-odd, $\sigma$-odd)"
                 "\n"
                 rf"$\Sigma>0$ median $={pos_med_e:+.3f}$; "
                 rf"rank-biserial vs $\Sigma=0$: ${r_rb:+.2f}$")
    ax.legend(fontsize=8.5, loc="upper left")
    ax.set_xlim(-1.05, 1.05)

    plt.tight_layout()
    plt.savefig(OUT, dpi=150)
    plt.close()
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
