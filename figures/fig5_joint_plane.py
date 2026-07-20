"""Regenerate Fig 5 (joint_plane.png): the (phi, psi) plane in log(1+x)
coordinates. The silent population is split into 33 robustly-silent (crosses)
and 5 rare-basin (open circles) rules per silent38_summary.csv.

"Silent" = phi>0 rules with G_star_alt < 1e-3 (main-sweep definition). The
G_star_alt column lives in composite_scores.csv; the robust/rare label lives
in silent38_summary.csv (label == "PROTOCOL-SILENT" flags the rare-basin
rules; the remainder are the robustly-silent ones).
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "joint_plane.png"

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

    cs["Sigma_n"] = pd.to_numeric(cs["Sigma"], errors="coerce")
    cs["N_sl_n"] = pd.to_numeric(cs["N_sl"], errors="coerce")
    cs["Gstar_n"] = pd.to_numeric(cs["G_star_alt"], errors="coerce")
    cs = cs.dropna(subset=["Sigma_n", "N_sl_n"]).copy()
    print(f"with numeric phi, psi: {len(cs)}")

    cs["logS"] = np.log1p(cs["Sigma_n"])
    cs["logL"] = np.log1p(cs["N_sl_n"])

    silent_mask = (cs["Sigma_n"] > 0) & (cs["Gstar_n"] < SILENT_THRESHOLD)
    n_silent = int(silent_mask.sum())
    print(f"silent rules (phi>0, G*_alt<{SILENT_THRESHOLD}): {n_silent}")

    stress = pd.read_csv(ROOT / "data" / "silent38_summary.csv")
    rare_pairs = set(map(tuple,
        stress[stress["label"] == "PROTOCOL-SILENT"][["B_mask", "S_mask"]].values.tolist()))
    silent_pairs = set(map(tuple,
        cs.loc[silent_mask, ["B_mask", "S_mask"]].values.tolist()))
    robust_pairs = silent_pairs - rare_pairs
    print(f"  robustly-silent (33 expected): {len(robust_pairs)}")
    print(f"  rare-basin      (5 expected):  {len(rare_pairs)}")
    robust_mask = silent_mask & cs.apply(
        lambda r: (int(r["B_mask"]), int(r["S_mask"])) in robust_pairs, axis=1)
    rare_mask = silent_mask & cs.apply(
        lambda r: (int(r["B_mask"]), int(r["S_mask"])) in rare_pairs, axis=1)

    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    rng = np.random.default_rng(0)
    jx = rng.uniform(-0.025, 0.025, size=len(cs))
    jy = rng.uniform(-0.025, 0.025, size=len(cs))

    sig0 = cs["Sigma_n"] == 0
    sigp_loud = (cs["Sigma_n"] > 0) & (~silent_mask)

    ax.scatter(cs.loc[sig0, "logS"] + jx[sig0.values],
               cs.loc[sig0, "logL"] + jy[sig0.values],
               s=2, alpha=0.18, color="0.55", rasterized=True,
               label=fr"$\phi=0$ ({int(sig0.sum())})")
    ax.scatter(cs.loc[sigp_loud, "logS"] + jx[sigp_loud.values],
               cs.loc[sigp_loud, "logL"] + jy[sigp_loud.values],
               s=10, alpha=0.7, color="crimson",
               edgecolor="black", linewidth=0.15,
               label=fr"$\phi>0$, $\mathcal{{G}}^*_\mathrm{{alt}}\geq{SILENT_THRESHOLD}$ ({int(sigp_loud.sum())})")
    ax.scatter(cs.loc[robust_mask, "logS"] + jx[robust_mask.values],
               cs.loc[robust_mask, "logL"] + jy[robust_mask.values],
               s=36, alpha=0.95, color="black", linewidth=1.0,
               marker="x",
               label=fr"robustly silent ($n={int(robust_mask.sum())}$)")
    ax.scatter(cs.loc[rare_mask, "logS"] + jx[rare_mask.values],
               cs.loc[rare_mask, "logL"] + jy[rare_mask.values],
               s=60, alpha=0.95, facecolor="none", edgecolor="black",
               linewidth=1.2, marker="o",
               label=fr"rare-basin ($n={int(rare_mask.sum())}$)")

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

    ax.set_xlabel(r"$\log(1 + \phi)$ — propagation, $B$-side")
    ax.set_ylabel(r"$\log(1 + \psi)$ — storage, $S$-side")
    ax.set_title(r"Joint propagation--storage plane $(\phi, \psi)$",
                 fontsize=12)
    ax.legend(loc="lower right", fontsize=8.5, framealpha=0.93)

    plt.tight_layout()
    plt.savefig(OUT, dpi=150)
    plt.close()
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
