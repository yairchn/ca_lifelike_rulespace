"""3-panel Figure 1 — coordinate triptych of all 262,144 rules.

Panels (shared layers): (lambda, mu), (delta, mu), (lambda, delta).
Each shows: grey cloud (all rules subsample), B=S 512 rules as distinct
marker, Sigma>0 424 rules as crimson dots, anchors as edge-only outlined.
Saves figures/binder_lambda_mu.png (overwrites).
"""
from __future__ import annotations
from math import comb
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FIG = ROOT / "outputs" / "binder_lambda_mu.png"
ENUM = ROOT / "data" / "spaceship_enumeration.csv"
MANIFEST = ROOT / "data" / "manifest_mu_full.parquet"

# Canonical coordinate block
C8 = [comb(8, n) for n in range(9)]
C7 = [comb(7, k) for k in range(8)]
pop = np.array([bin(m).count("1") for m in range(512)])
centre = np.array([sum(C8[n] for n in range(9) if (m >> n) & 1) / 256.0
                   for m in range(512)])
periph = np.array([sum(C7[k] for k in range(8)
                       if ((m >> k) & 1) != ((m >> (k + 1)) & 1)) / 256.0
                   for m in range(512)])


def coords(B, S):
    lam = (pop[B] + pop[S]) / 18.0
    delta = (pop[B] - pop[S]) / 9.0
    mu_centre = centre[B ^ S]
    mu = (mu_centre + 8 * (periph[B] + periph[S])) / 9.0
    return lam, delta, mu, mu_centre


# Build full 262144-rule table
B_all = np.arange(262144) & 0x1FF
S_all = (np.arange(262144) >> 9) & 0x1FF
lam_all, delta_all, mu_all, muc_all = coords(B_all, S_all)
print(f"computed all 262144 rules; |B=S|={(B_all==S_all).sum()}")

# Cross-check mu vs manifest
if MANIFEST.exists():
    man = pd.read_parquet(MANIFEST)
    df = pd.DataFrame({"B_mask": B_all, "S_mask": S_all, "mu_calc": mu_all})
    df = df.merge(man[["B_mask", "S_mask", "mu"]], on=["B_mask", "S_mask"])
    print(f"max |mu_calc - mu_manifest| = {(df['mu_calc'] - df['mu']).abs().max():.3e}")

# Masks
bs_mask = (B_all == S_all)
enum = pd.read_csv(ENUM)
sig_rids = set(
    (int(s) << 9) | int(b)
    for b, s in zip(enum[enum["Sigma"] > 0]["B_mask"].values,
                    enum[enum["Sigma"] > 0]["S_mask"].values)
)
sig_mask = np.array([r in sig_rids for r in range(262144)])
print(f"|Sigma>0|={sig_mask.sum()}; B=S ∩ Sigma>0 = {(bs_mask & sig_mask).sum()}")

# Anchors
ANCHORS = {
    "Conway":     ({3}, {2, 3}, "IV"),
    "HighLife":   ({3, 6}, {2, 3}, "IV"),
    "2x2":        ({3, 6}, {1, 2, 5}, "IV"),
    "Day & Night":({3, 6, 7, 8}, {3, 4, 6, 7, 8}, "IV"),
    "Seeds":      ({2}, set(), "III"),
    "Replicator": ({1, 3, 5, 7}, {1, 3, 5, 7}, "III"),
    "Maze":       ({3}, {1, 2, 3, 4, 5}, "II"),
}

def set_to_mask(s):
    m = 0
    for n in s:
        m |= 1 << n
    return m

anchor_rows = []
for nm, (Bset, Sset, cls) in ANCHORS.items():
    B = set_to_mask(Bset); S = set_to_mask(Sset)
    lam, delta, mu, muc = coords(np.array([B]), np.array([S]))
    anchor_rows.append({"name": nm, "B": B, "S": S, "class": cls,
                        "lambda": float(lam[0]),
                        "delta": float(delta[0]),
                        "mu": float(mu[0]),
                        "mu_centre": float(muc[0])})
anchor_df = pd.DataFrame(anchor_rows)
print("\nAnchor coordinates:")
print(anchor_df.to_string(index=False))

# Verify the five required numbers
assert abs(anchor_df.loc[anchor_df["name"]=="Conway", "mu"].iloc[0] - 0.3524) < 1e-3
assert abs(anchor_df.loc[anchor_df["name"]=="HighLife", "delta"].iloc[0]) < 1e-9
assert abs(anchor_df.loc[anchor_df["name"]=="HighLife", "mu"].iloc[0] - 0.4618) < 1e-3
assert abs(anchor_df.loc[anchor_df["name"]=="Replicator", "delta"].iloc[0]) < 1e-9
assert abs(anchor_df.loc[anchor_df["name"]=="Replicator", "mu"].iloc[0] - 0.8889) < 1e-3
print("\nAll five verification checks PASSED.")

# ---- Build the figure ----
CLASS_OUTLINE = {"IV": "#cc6600", "III": "#1a8a1a", "II": "#0066cc"}
CLASS_MARKER  = {"IV": "o", "III": "^", "II": "s"}
BS_COLOR = "#1f4ea1"     # royal blue
BS_MARKER = "x"          # thin line X — distinct from crimson dot

rng = np.random.default_rng(0)
bg_idx = rng.choice(262144, size=16000, replace=False)
# Tiny jitter for visibility on the discrete grid
jx_b = rng.uniform(-0.012, 0.012, size=len(bg_idx))
jy_b = rng.uniform(-0.012, 0.012, size=len(bg_idx))
# Jitter for B=S points (especially in (delta,mu) where they pile on delta=0)
bs_idx = np.where(bs_mask)[0]
jx_bs = rng.uniform(-0.015, 0.015, size=len(bs_idx))
jy_bs = rng.uniform(-0.015, 0.015, size=len(bs_idx))

fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.4))

def draw(ax, x_all, y_all, x_label, y_label, title,
         anchor_xcol, anchor_ycol, x_lim, y_lim,
         jitter_bs_x=True):
    # Layer 1: grey cloud
    ax.scatter(x_all[bg_idx] + jx_b, y_all[bg_idx] + jy_b,
               s=2, alpha=0.35, color="0.6", rasterized=True)
    # Layer 2: B=S 512 — thin X strokes, no edge box
    bx = x_all[bs_idx] + (jx_bs if jitter_bs_x else 0)
    by = y_all[bs_idx]
    ax.scatter(bx, by, s=22, marker=BS_MARKER, color=BS_COLOR,
               linewidths=0.35, edgecolor="none",
               alpha=0.85, zorder=4)
    # Layer 3: Sigma>0 424
    ax.scatter(x_all[sig_mask], y_all[sig_mask],
               s=5, color="crimson",
               edgecolor="black", linewidth=0.12,
               alpha=0.9, zorder=3)
    # Layer 4: anchors — outlined (edge-only)
    for _, r in anchor_df.iterrows():
        cls = r["class"]
        ax.scatter([r[anchor_xcol]], [r[anchor_ycol]],
                   s=120, marker=CLASS_MARKER[cls],
                   facecolor="white",
                   edgecolor=CLASS_OUTLINE[cls],
                   linewidth=2.0, zorder=5)
        ax.annotate(r["name"], (r[anchor_xcol], r[anchor_ycol]),
                    xytext=(6, 4), textcoords="offset points",
                    fontsize=8.0, fontweight="bold", zorder=6)
    ax.set_xlabel(x_label, fontsize=10.5)
    ax.set_ylabel(y_label, fontsize=10.5)
    ax.set_title(title, fontsize=11)
    ax.set_xlim(*x_lim)
    ax.set_ylim(*y_lim)

# Panel 1: (lambda, mu)
draw(axes[0], lam_all, mu_all,
     r"Langton activity $\lambda$", r"average sensitivity $\mu$",
     r"$(\lambda,\mu)$",
     "lambda", "mu",
     (0, 1), (-0.02, 1.0))

# Panel 2: (delta, mu)
draw(axes[1], delta_all, mu_all,
     r"asymmetry $\delta$", r"average sensitivity $\mu$",
     r"$(\delta,\mu)$",
     "delta", "mu",
     (-1, 1), (-0.02, 1.0))
axes[1].axvline(0, color="0.3", lw=0.4, ls=":", alpha=0.5)
# annotate HighLife/Replicator δ=0 caveat
hl = anchor_df[anchor_df["name"] == "HighLife"].iloc[0]
axes[1].annotate(
    r"$\delta=0$ but $B \neq S$" "\n" r"(equal cardinality, not totalistic)",
    xy=(hl["delta"], hl["mu"]),
    xytext=(0.36, 0.30),
    textcoords="data",
    fontsize=7.5,
    color="#333",
    bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
              edgecolor="#555", alpha=0.95, linewidth=0.5),
    arrowprops=dict(arrowstyle="->", color="#444", lw=0.7,
                    connectionstyle="arc3,rad=-0.2"))

# Panel 3: (lambda, delta)
draw(axes[2], lam_all, delta_all,
     r"Langton activity $\lambda$", r"asymmetry $\delta$",
     r"$(\lambda,\delta)$",
     "lambda", "delta",
     (0, 1), (-1, 1))
axes[2].axhline(0, color="0.3", lw=0.4, ls=":", alpha=0.5)

# Shared legend (bottom). Deduplicate by label so no anchor class can appear twice.
_raw_handles = [
    Line2D([0], [0], marker='o', color='none', markerfacecolor="0.6",
           markersize=4.5, label="all rules (subsample of 262,144)"),
    Line2D([0], [0], marker=BS_MARKER, color=BS_COLOR,
           markersize=8, markeredgewidth=0.35, linestyle='none',
           label=r"$B=S$ (512), $\mu_\mathrm{centre}=0$"),
    Line2D([0], [0], marker='o', color='none', markerfacecolor="crimson",
           markeredgecolor='black', markersize=5,
           label=r"$\Sigma>0$ (424)"),
    Line2D([0], [0], marker='o', color='none', markerfacecolor='white',
           markeredgecolor=CLASS_OUTLINE["IV"], markeredgewidth=1.8,
           markersize=9, label="Class IV anchor"),
    Line2D([0], [0], marker='^', color='none', markerfacecolor='white',
           markeredgecolor=CLASS_OUTLINE["III"], markeredgewidth=1.8,
           markersize=9, label="Class III anchor"),
    Line2D([0], [0], marker='s', color='none', markerfacecolor='white',
           markeredgecolor=CLASS_OUTLINE["II"], markeredgewidth=1.8,
           markersize=9, label="Class II anchor"),
]
_seen = set()
legend_handles = []
for h in _raw_handles:
    lbl = h.get_label()
    if lbl in _seen:
        continue
    _seen.add(lbl)
    legend_handles.append(h)
fig.legend(handles=legend_handles, loc="lower center", ncol=len(legend_handles),
           fontsize=8.5, frameon=True, framealpha=0.95,
           bbox_to_anchor=(0.5, -0.02))

plt.tight_layout(rect=[0, 0.04, 1, 1])
plt.savefig(FIG, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nwrote {FIG}")
print(f"\nB=S marker style: '{BS_MARKER}' in {BS_COLOR} (royal blue), "
      f"s=22, linewidths=0.35, no edge")
