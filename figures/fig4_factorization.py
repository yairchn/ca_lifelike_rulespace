"""Fig 4 (final): 2-panel factorization figure.

Panel (a): 3-bar chart of birth-set-support cardinalities at threshold k>=4,
  showing the strict nesting B_prop=24 subset B_osc=128 subset B_sl=176 (of 512
  possible birth sets).
Panel (b): #SL vs |S| scatter, split by B in B_prop (orange) vs. outside (grey);
  Spearman rho = +0.555 inside vs +0.135 outside.

Overwrites manuscript/figures/factorization.png.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
OUT  = ROOT / "manuscript" / "figures" / "factorization.png"


def popc(m): return bin(int(m)).count("1")


print("loading ...")
sh  = pd.read_csv(ROOT / "data" / "spaceship_enumeration.csv")[["B_mask","S_mask","Sigma"]]
sl  = pd.read_parquet(ROOT / "data" / "still_life_enumeration.parquet")[
        ["B_mask","S_mask","still_life_orbits"]]
osc = pd.read_csv(ROOT / "data" / "oscillator_enumeration.csv")[["B_mask","S_mask","Omega"]]
df = sh.merge(sl, on=["B_mask","S_mask"]).merge(osc, on=["B_mask","S_mask"])
print(f"  merged {len(df):,} rules")

# Per-B companion counts (>=1 orbit representative)
N_sl = df.groupby("B_mask", as_index=True).apply(
    lambda x: int((x["still_life_orbits"] > 0).sum()),
    include_groups=False).reindex(range(512), fill_value=0)
N_om = df.groupby("B_mask", as_index=True).apply(
    lambda x: int((x["Omega"] > 0).sum()),
    include_groups=False).reindex(range(512), fill_value=0)
N_sg = df.groupby("B_mask", as_index=True).apply(
    lambda x: int((x["Sigma"] > 0).sum()),
    include_groups=False).reindex(range(512), fill_value=0)

# Paper's threshold: k >= 4 companions
K = 4
B_sl_set   = set(N_sl.index[N_sl >= K])
B_osc_set  = set(N_om.index[N_om >= K])
B_prop_set = set(N_sg.index[N_sg >= K])
print(f"  |B_sl|   = {len(B_sl_set)}")
print(f"  |B_osc|  = {len(B_osc_set)}")
print(f"  |B_prop| = {len(B_prop_set)}")
assert B_prop_set <= B_osc_set <= B_sl_set

# Sanity: any-support count for the caption note.
B_prop_any = set(N_sg.index[N_sg >= 1])
print(f"  |B_prop| at N>=1 (for caption footnote): {len(B_prop_any)}")

# Spearman rho(#SL, |S|) inside vs outside B_prop
df["S_card"] = df["S_mask"].apply(popc)
inP  = df[df["B_mask"].isin(B_prop_set)]
outP = df[~df["B_mask"].isin(B_prop_set)]
rho_in,  _ = spearmanr(inP["still_life_orbits"], inP["S_card"])
rho_out, _ = spearmanr(outP["still_life_orbits"], outP["S_card"])
print(f"  Spearman rho(#SL,|S|): inside={rho_in:+.3f}, outside={rho_out:+.3f}")

# ---- FIGURE ----
fig, axes = plt.subplots(1, 2, figsize=(13, 5.0),
                          gridspec_kw=dict(wspace=0.28))

# --- Panel A: 3-bar chart of nested cardinalities ---
axA = axes[0]
labels = [r"$\mathcal{B}_\mathrm{sl}$",
          r"$\mathcal{B}_\mathrm{osc}$",
          r"$\mathcal{B}_\mathrm{prop}$"]
vals   = [len(B_sl_set), len(B_osc_set), len(B_prop_set)]
colors = ["#888888", "#2e7dc7", "#e76f51"]
bars = axA.bar(labels, vals, color=colors, edgecolor="black",
                linewidth=0.6, alpha=0.92, width=0.62)
for b, v in zip(bars, vals):
    axA.text(b.get_x() + b.get_width()/2, v + 12, str(v),
             ha="center", va="bottom", fontsize=13, fontweight="bold")
axA.axhline(512, color="0.55", ls=":", lw=0.9)
axA.text(2.42, 512, "$512$ = all birth sets", ha="right", va="bottom",
         fontsize=10, color="0.35")
axA.set_ylabel("number of birth sets $B$ with "
               r"$\geq 4$ survival companions", fontsize=11)
axA.set_ylim(0, 560)
axA.set_title(
    r"A — birth-set support:  "
    rf"$\mathcal{{B}}_\mathrm{{prop}}\;({len(B_prop_set)}) \subset "
    rf"\mathcal{{B}}_\mathrm{{osc}}\;({len(B_osc_set)}) \subset "
    rf"\mathcal{{B}}_\mathrm{{sl}}\;({len(B_sl_set)})$",
    fontsize=11.5)
axA.spines[["top","right"]].set_visible(False)
axA.tick_params(axis="x", labelsize=13)

# --- Panel B: #SL vs |S| scatter ---
axB = axes[1]
sub_in  = inP.dropna(subset=["still_life_orbits", "S_card"]).sample(
            min(10000, len(inP)),  random_state=0)
sub_out = outP.dropna(subset=["still_life_orbits", "S_card"]).sample(
            min(20000, len(outP)), random_state=0)
rng0 = np.random.RandomState(0)
rng1 = np.random.RandomState(1)
axB.scatter(sub_out["S_card"] + rng0.uniform(-0.18, 0.18, len(sub_out)),
            sub_out["still_life_orbits"], s=3, alpha=0.10,
            color="#888",
            label=fr"outside $\mathcal{{B}}_\mathrm{{prop}}$  ($\rho={rho_out:+.3f}$)")
axB.scatter(sub_in["S_card"] + rng1.uniform(-0.18, 0.18, len(sub_in)),
            sub_in["still_life_orbits"], s=4, alpha=0.30,
            color="#e76f51",
            label=fr"inside $\mathcal{{B}}_\mathrm{{prop}}$  ($\rho={rho_in:+.3f}$)")
axB.set_xlabel(r"$|S|$  (survival-set cardinality)")
axB.set_ylabel(r"$\#$SL  (still-life orbits per rule)")
axB.set_title(r"B — storage scales with $|S|$ inside the propagation gate",
              fontsize=11.5)
axB.legend(loc="upper left", fontsize=10)

plt.tight_layout()
plt.savefig(OUT, dpi=140)
plt.close()
print(f"[fig] wrote {OUT}")
