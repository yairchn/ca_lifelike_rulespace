"""Regenerate Fig 5 (joint_plane.png): the (Sigma, #SL) plane in log(1+x)
coordinates. Mark the 38 glider-capable-silent rules. No "F3" title, no
$\bar R_g$ colour axis — only items the main text uses.

We mark "silent" as Sigma>0 rules with G_star_alt < 1e-3 (the manuscript's
definition). The G_star_alt column lives in composite_scores.csv.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "manuscript" / "figures" / "joint_plane.png"

SILENT_THRESHOLD = 1e-3   # G_star_alt < this means "silent"

ANCHORS = {
    "Conway":     (8, 12),
    "HighLife":   (72, 12),
    "2x2":        (72, 38),
    "Day & Night":(456, 472),
    "Seeds":      (4, 0),
    "Replicator": (170, 170),
    "Maze":       (8, 62),
    "Move":       (40, 124),
}


def main():
    cs = pd.read_csv(ROOT / "data" / "composite_scores.csv", low_memory=False)
    print(f"loaded {len(cs)} rules from composite_scores.csv")

    # Ensure Sigma column is numeric (some rows are header-row artefacts)
    cs["Sigma_n"] = pd.to_numeric(cs["Sigma"], errors="coerce")
    cs["N_sl_n"] = pd.to_numeric(cs["N_sl"], errors="coerce")
    cs["Gstar_n"] = pd.to_numeric(cs["G_star_alt"], errors="coerce")
    cs = cs.dropna(subset=["Sigma_n", "N_sl_n"]).copy()
    print(f"with numeric Σ, #SL: {len(cs)}")

    cs["logS"] = np.log1p(cs["Sigma_n"])
    cs["logL"] = np.log1p(cs["N_sl_n"])

    # silent population
    silent_mask = (cs["Sigma_n"] > 0) & (cs["Gstar_n"] < SILENT_THRESHOLD)
    n_silent = int(silent_mask.sum())
    print(f"silent rules (Sigma>0, G*_alt<{SILENT_THRESHOLD}): {n_silent}")

    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    rng = np.random.default_rng(0)
    # tiny jitter to spread overlapping integer-valued log1p points
    jx = rng.uniform(-0.025, 0.025, size=len(cs))
    jy = rng.uniform(-0.025, 0.025, size=len(cs))

    # background: all rules
    bg = (cs["Sigma_n"] == 0) | (~silent_mask & (cs["Sigma_n"] > 0))
    bg_all = ~silent_mask          # everything not in the silent set
    sig0 = cs["Sigma_n"] == 0
    sigp_loud = (cs["Sigma_n"] > 0) & (~silent_mask)

    ax.scatter(cs.loc[sig0, "logS"] + jx[sig0.values],
               cs.loc[sig0, "logL"] + jy[sig0.values],
               s=2, alpha=0.18, color="0.55", rasterized=True,
               label=fr"$\Sigma=0$ ({int(sig0.sum())})")
    ax.scatter(cs.loc[sigp_loud, "logS"] + jx[sigp_loud.values],
               cs.loc[sigp_loud, "logL"] + jy[sigp_loud.values],
               s=10, alpha=0.7, color="crimson",
               edgecolor="black", linewidth=0.15,
               label=fr"$\Sigma>0$, $\mathcal{{G}}^*_\mathrm{{alt}}\geq{SILENT_THRESHOLD}$ ({int(sigp_loud.sum())})")
    ax.scatter(cs.loc[silent_mask, "logS"] + jx[silent_mask.values],
               cs.loc[silent_mask, "logL"] + jy[silent_mask.values],
               s=32, alpha=0.95, color="black", linewidth=0.9,
               marker="x",
               label=fr"glider-capable-silent ($n={n_silent}$)")

    # anchors
    for nm, (B, S) in ANCHORS.items():
        h = cs[(cs["B_mask"] == B) & (cs["S_mask"] == S)]
        if not len(h):
            continue
        h = h.iloc[0]
        x = float(h["logS"]); y = float(h["logL"])
        ax.scatter([x], [y], s=120, marker="o",
                   color="gold", edgecolor="black", linewidth=0.9, zorder=5)
        ax.annotate(nm, (x, y), xytext=(7, 4),
                    textcoords="offset points", fontsize=9.5,
                    fontweight="bold", zorder=6)

    ax.set_xlabel(r"$\log(1 + \Sigma)$ — propagation, $B$-side")
    ax.set_ylabel(r"$\log(1 + \#\mathrm{SL})$ — storage, $S$-side")
    ax.set_title(r"Joint propagation--storage plane $(\Sigma, \#\mathrm{SL})$",
                 fontsize=12)
    ax.legend(loc="lower right", fontsize=8.5, framealpha=0.93)

    plt.tight_layout()
    plt.savefig(OUT, dpi=150)
    plt.close()
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
