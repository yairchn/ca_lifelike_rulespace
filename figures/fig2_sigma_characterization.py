"""Figure 2 — Sigma characterisation, 4 panels.

  (a) Named vs unnamed (2 vs 422 in our canonical list).
  (b) Velocity class, diagonal vs orthogonal (560 vs 316), per-ship.
  (c) Period distribution, mode T=4.
  (d) NEW: (mu_centre, |B|) plane, points = the 424 Sigma>0 rules, coloured by
      direction class (diag-only / orth-only). Four-corner orthogonal-fraction
      numbers overlaid at the medians.
"""
from __future__ import annotations
import ast
import json
from math import comb
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
ENUM = ROOT / "data" / "spaceship_enumeration.csv"
DETAIL = ROOT / "data" / "spaceships_detail.jsonl"
COMPOSITE = ROOT / "data" / "composite_scores.csv"
MANIFEST = ROOT / "data" / "manifest_mu_full.parquet"
OUT = ROOT / "manuscript" / "figures" / "sigma_characterization.png"

# Canonical coordinate block
C8 = [comb(8, n) for n in range(9)]
C7 = [comb(7, k) for k in range(8)]
pop = np.array([bin(m).count("1") for m in range(512)])
centre = np.array([sum(C8[n] for n in range(9) if (m >> n) & 1) / 256.0
                   for m in range(512)])


def classify_velocity(v):
    dx, dy = int(v[0]), int(v[1])
    ax, ay = abs(dx), abs(dy)
    if (dx, dy) == (0, 0):
        return "zero"
    if ax == ay:
        return "diag"
    if (ax == 0) ^ (ay == 0):
        return "orth"
    return "knight"


def rule_direction(velocities):
    cs = {classify_velocity(v) for v in velocities}
    cs.discard("zero")
    if "knight" in cs:
        return "knight"
    if "diag" in cs and "orth" in cs:
        return "both"
    if "diag" in cs:
        return "diag-only"
    if "orth" in cs:
        return "orth-only"
    return "unknown"


def main():
    e = pd.read_csv(ENUM)
    pos = e[e["Sigma"] > 0].copy().reset_index(drop=True)
    print(f"Sigma>0: {len(pos)}")

    # direction class per rule
    pos["dir_class"] = pos["velocities_found"].apply(
        lambda s: rule_direction(ast.literal_eval(s) if isinstance(s, str) else []))
    print("direction class counts:")
    print(pos["dir_class"].value_counts())

    # named vs unnamed from composite_scores
    cs = pd.read_csv(COMPOSITE, low_memory=False)
    named_set = set()
    for _, r in cs.iterrows():
        name = r["name"]
        if isinstance(name, str) and name and name != "nan":
            named_set.add((int(r["B_mask"]), int(r["S_mask"])))
    pos["named"] = pos.apply(
        lambda r: (int(r["B_mask"]), int(r["S_mask"])) in named_set, axis=1)
    n_named = int(pos["named"].sum())
    n_unnamed = int(len(pos) - n_named)
    print(f"named: {n_named}, unnamed: {n_unnamed}")

    # velocity classification: per-ship via detail file
    n_diag = n_orth = n_knight = 0
    with open(DETAIL) as f:
        for line in f:
            d = json.loads(line)
            dx = d.get("dx"); dy = d.get("dy")
            if dx is None or dy is None:
                continue
            ax, ay = abs(int(dx)), abs(int(dy))
            if ax == ay and ax > 0:
                n_diag += 1
            elif (ax == 0) ^ (ay == 0):
                n_orth += 1
            else:
                n_knight += 1
    print(f"per-ship velocities: diag={n_diag}, orth={n_orth}, knight={n_knight}")

    # period distribution
    periods = pos["min_period_found"].dropna().astype(int)
    period_mode = periods.mode().iloc[0]
    print(f"period mode T={period_mode}")

    # (mu_centre, |B|) attribution
    pos["|B|"] = pos["B_mask"].apply(lambda x: bin(int(x)).count("1"))
    pos["mu_centre"] = pos.apply(
        lambda r: centre[int(r["B_mask"]) ^ int(r["S_mask"])], axis=1)
    pure = pos[pos["dir_class"].isin(["diag-only", "orth-only"])].copy()
    print(f"pure (diag/orth): {len(pure)} "
          f"(diag={(pure['dir_class']=='diag-only').sum()}, "
          f"orth={(pure['dir_class']=='orth-only').sum()})")

    # four-corner orthogonal fractions
    B_med = pure["|B|"].median()
    muc_med = pure["mu_centre"].median()
    print(f"medians: |B|={B_med}, mu_centre={muc_med:.3f}")
    corners = {}
    for B_lbl, B_sel in [("small|B|", pure["|B|"] <= B_med),
                          ("large|B|", pure["|B|"] > B_med)]:
        for mu_lbl, mu_sel in [("low mu_c", pure["mu_centre"] <= muc_med),
                                ("high mu_c", pure["mu_centre"] > muc_med)]:
            cell = pure[B_sel & mu_sel]
            n = len(cell)
            pct = 100 * (cell["dir_class"] == "orth-only").mean() if n else 0
            corners[(B_lbl, mu_lbl)] = (n, pct)
            print(f"  {B_lbl} × {mu_lbl}: n={n}, orth={pct:.1f}%")

    # ---- build figure ----
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 9.5))

    # (a) named vs unnamed
    ax = axes[0, 0]
    bars = ax.bar(["named", "unnamed"], [n_named, n_unnamed],
                  color=["#2a9d8f", "#e76f51"], width=0.6,
                  edgecolor="black", linewidth=0.5)
    for b, v in zip(bars, [n_named, n_unnamed]):
        ax.text(b.get_x() + b.get_width() / 2, v + 6,
                str(v), ha="center", va="bottom",
                fontsize=12, fontweight="bold")
    ax.set_ylabel(r"rules with $\Sigma>0$")
    ax.set_ylim(0, max(n_named, n_unnamed) * 1.1)
    ax.set_title(rf"(a) named vs. unnamed ({n_named} of {len(pos)} named)")

    # (b) velocity class
    ax = axes[0, 1]
    bars = ax.bar(["diagonal", "orthogonal"], [n_diag, n_orth],
                  color=["#1f4ea1", "#c0392b"], width=0.6,
                  edgecolor="black", linewidth=0.5)
    for b, v in zip(bars, [n_diag, n_orth]):
        ax.text(b.get_x() + b.get_width() / 2, v + 8,
                str(v), ha="center", va="bottom",
                fontsize=12, fontweight="bold")
    ax.set_ylabel(r"spaceship occurrences")
    ax.set_ylim(0, max(n_diag, n_orth) * 1.12)
    ax.set_title(rf"(b) velocity class ({n_diag} vs. {n_orth})")

    # (c) period distribution
    ax = axes[1, 0]
    hist = periods.value_counts().sort_index()
    Trange = np.arange(2, 17)
    counts = [int(hist.get(t, 0)) for t in Trange]
    bars = ax.bar(Trange, counts, color="#5b8def", width=0.85,
                  edgecolor="black", linewidth=0.4)
    ax.set_xlabel(r"min period $T$")
    ax.set_ylabel(r"rules with $\Sigma>0$")
    ax.set_xticks(Trange)
    ax.set_title(rf"(c) period distribution (mode $T={period_mode}$)")

    # (d) (mu_centre, |B|) by direction with corner annotations
    ax = axes[1, 1]
    rng = np.random.default_rng(0)
    diag = pure[pure["dir_class"] == "diag-only"]
    orth = pure[pure["dir_class"] == "orth-only"]
    j_diag = rng.uniform(-0.18, 0.18, size=len(diag))
    j_orth = rng.uniform(-0.18, 0.18, size=len(orth))
    ax.scatter(diag["mu_centre"], diag["|B|"] + j_diag,
               s=30, color="#1f4ea1", alpha=0.75, marker="o",
               edgecolor="black", linewidth=0.2,
               label=fr"diagonal-only ({len(diag)})")
    ax.scatter(orth["mu_centre"], orth["|B|"] + j_orth,
               s=30, color="#c0392b", alpha=0.75, marker="s",
               edgecolor="black", linewidth=0.2,
               label=fr"orthogonal-only ({len(orth)})")
    ax.axvline(muc_med, color="0.4", lw=0.6, ls="--", alpha=0.6)
    ax.axhline(B_med, color="0.4", lw=0.6, ls="--", alpha=0.6)

    # four corner labels at fixed positions
    # axis limits are roughly mu_centre [0, 1], |B| [1, 8]
    xlim_l = 0.0; xlim_h = 1.02
    ylim_l = 0.3; ylim_h = 8.6

    def annotate_corner(cx, cy, label, pct):
        bg = "white" if pct < 50 else "#ffd6d6"
        ax.annotate(f"{label}\n{pct:.0f}% orth",
                    xy=(cx, cy), ha="center", va="center",
                    fontsize=8.5, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.3",
                              facecolor=bg, edgecolor="0.4",
                              linewidth=0.6, alpha=0.92))

    # corners in display order: small|B|/low μc (LL), small|B|/high μc (LR),
    # large|B|/low μc (UL), large|B|/high μc (UR)
    n_ll, p_ll = corners[("small|B|", "low mu_c")]
    n_lr, p_lr = corners[("small|B|", "high mu_c")]
    n_ul, p_ul = corners[("large|B|", "low mu_c")]
    n_ur, p_ur = corners[("large|B|", "high mu_c")]

    # Place labels at quartile-ish positions
    cx_lo = muc_med * 0.45
    cx_hi = muc_med + (xlim_h - muc_med) * 0.55
    cy_lo = (ylim_l + B_med) / 2.0
    cy_hi = (B_med + ylim_h) / 2.0

    annotate_corner(cx_lo, cy_lo, f"n={n_ll}", p_ll)
    annotate_corner(cx_hi, cy_lo, f"n={n_lr}", p_lr)
    annotate_corner(cx_lo, cy_hi, f"n={n_ul}", p_ul)
    annotate_corner(cx_hi, cy_hi, f"n={n_ur}", p_ur)

    ax.set_xlim(xlim_l, xlim_h)
    ax.set_ylim(ylim_l, ylim_h)
    ax.set_yticks(range(1, 9))
    ax.set_xlabel(r"$\mu_\mathrm{centre}$")
    ax.set_ylabel(r"$|B|$")
    ax.set_title("(d) direction by $(\\mu_\\mathrm{centre},\\,|B|)$:  "
                 "orth requires both levers high")
    ax.legend(loc="upper left", fontsize=8.5, framealpha=0.93)

    plt.tight_layout()
    plt.savefig(OUT, dpi=150)
    plt.close()
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
