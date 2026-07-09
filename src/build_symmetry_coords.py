"""Numerical verification of the (T, σ) parity organisation of the rule-table
coordinates on ALL 262144 life-like rules. STOP and report on any failure.

Coordinates:
  lambda      = (|B|+|S|) / 18
  Delta       = (|B|-|S|) /  9
  mu_centre   = (1/256) sum_n C(8,n) [n in B △ S]
  mu_periph   = (1/256) sum_k C(7,k) ([k in B△(B-1)] + [k in S△(S-1)])    edge indicator at k
  mu          = (1/9)(mu_centre + 8 mu_periph)
  Delta_edge  = (e(B) - e(S)) / 8,   where e(M) = #{k in 0..7: 1[k in M] != 1[k+1 in M]}

Involutions:
  T:(B,S) -> (S,B)
  sigma:(B,S) -> (not(rev(S)), not(rev(B))),  rev: bit k <-> bit 8-k,  not: 9-bit complement

Predicted parities:
   coordinate        T-action       sigma-action
   lambda            invariant      lambda -> 1 - lambda
   Delta             -> -Delta      invariant
   mu_centre         invariant      invariant
   mu_periph         invariant      invariant
   mu                invariant      invariant
   Delta_edge        -> -Delta_edge -> -Delta_edge
"""
from __future__ import annotations
import math
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "data"

def popcount9(m: int) -> int:
    return bin(m & 0x1FF).count("1")

def rev9(m: int) -> int:
    """Bit reversal in 9-bit mask: bit k -> bit 8-k."""
    r = 0
    for k in range(9):
        if (m >> k) & 1:
            r |= 1 << (8 - k)
    return r

def comp9(m: int) -> int:
    return (~m) & 0x1FF

def T(rid18: int) -> int:
    S = (rid18 >> 9) & 0x1FF
    B = rid18 & 0x1FF
    return (B << 9) | S

def sigma(rid18: int) -> int:
    S = (rid18 >> 9) & 0x1FF
    B = rid18 & 0x1FF
    new_B = comp9(rev9(S))
    new_S = comp9(rev9(B))
    return (new_S << 9) | new_B

def coords(B_mask: int, S_mask: int):
    bcard = popcount9(B_mask); scard = popcount9(S_mask)
    lam = (bcard + scard) / 18.0
    delta = (bcard - scard) / 9.0
    mu_c = sum(math.comb(8, n) for n in range(9)
               if ((B_mask >> n) & 1) ^ ((S_mask >> n) & 1)) / 256.0
    mu_p = 0
    for k in range(8):
        if ((B_mask >> k) & 1) ^ ((B_mask >> (k+1)) & 1):
            mu_p += math.comb(7, k)
        if ((S_mask >> k) & 1) ^ ((S_mask >> (k+1)) & 1):
            mu_p += math.comb(7, k)
    mu_p /= 256.0
    mu = (mu_c + 8 * mu_p) / 9.0
    # edge count
    def edges(M):
        return sum(1 for k in range(8) if ((M >> k) & 1) ^ ((M >> (k+1)) & 1))
    delta_edge = (edges(B_mask) - edges(S_mask)) / 8.0
    return lam, delta, mu_c, mu_p, mu, delta_edge


def main():
    print("Computing coordinates for all 262144 rules...")
    cols = []
    for rid18 in range(262144):
        S = (rid18 >> 9) & 0x1FF
        B = rid18 & 0x1FF
        cols.append((rid18, B, S) + coords(B, S))
    df = pd.DataFrame(cols, columns=["rid18","B","S","lam","delta","mu_c","mu_p","mu","delta_edge"])

    # Pre-build map rid -> coords for fast lookup
    print("Building T and sigma lookup arrays...")
    rid_to_idx = {r: i for i, r in enumerate(df["rid18"].values)}
    T_rid = np.array([T(r) for r in df["rid18"].values])
    sigma_rid = np.array([sigma(r) for r in df["rid18"].values])
    T_idx = np.array([rid_to_idx[r] for r in T_rid])
    sigma_idx = np.array([rid_to_idx[r] for r in sigma_rid])

    # Build predicted-after-action and compare
    results = []
    TOL = 1e-12

    def check(name, after, predicted):
        diff = np.max(np.abs(after - predicted))
        ok = diff < TOL
        results.append((name, "PASS" if ok else "FAIL", float(diff)))
        return ok

    arr = {c: df[c].values for c in ("lam","delta","mu_c","mu_p","mu","delta_edge")}

    # --- T action ---
    check("lam   under T    (predicted: invariant)",     arr["lam"][T_idx],        arr["lam"])
    check("delta under T    (predicted: -delta)",        arr["delta"][T_idx],      -arr["delta"])
    check("mu_c  under T    (predicted: invariant)",     arr["mu_c"][T_idx],       arr["mu_c"])
    check("mu_p  under T    (predicted: invariant)",     arr["mu_p"][T_idx],       arr["mu_p"])
    check("mu    under T    (predicted: invariant)",     arr["mu"][T_idx],         arr["mu"])
    check("d_edg under T    (predicted: -d_edge)",       arr["delta_edge"][T_idx], -arr["delta_edge"])

    # --- sigma action ---
    check("lam   under sigma (predicted: 1 - lam)",      arr["lam"][sigma_idx],        1.0 - arr["lam"])
    check("delta under sigma (predicted: invariant)",    arr["delta"][sigma_idx],      arr["delta"])
    check("mu_c  under sigma (predicted: invariant)",    arr["mu_c"][sigma_idx],       arr["mu_c"])
    check("mu_p  under sigma (predicted: invariant)",    arr["mu_p"][sigma_idx],       arr["mu_p"])
    check("mu    under sigma (predicted: invariant)",    arr["mu"][sigma_idx],         arr["mu"])
    check("d_edg under sigma (predicted: -d_edge)",      arr["delta_edge"][sigma_idx], -arr["delta_edge"])

    # --- T*T = identity, sigma*sigma = identity (involution checks)
    check("T involution            ",   df["rid18"].values[T_idx][T_idx].astype(float),    df["rid18"].values.astype(float))
    check("sigma involution        ",   df["rid18"].values[sigma_idx][sigma_idx].astype(float), df["rid18"].values.astype(float))

    print()
    print(f"{'check':55s} {'status':6s}  max|diff|")
    print("-" * 80)
    all_pass = True
    for name, status, diff in results:
        flag = " " if status == "PASS" else "!!"
        print(f"{flag} {name:55s} {status:6s}  {diff:.3e}")
        if status != "PASS":
            all_pass = False
    print()

    if not all_pass:
        print("PARITY VERIFICATION FAILED — STOPPING. Do not write the table.")
        return 1

    print("ALL PARITIES VERIFIED on 262,144 rules.")

    # --- Sigma>0 connection to Delta_edge ---
    print("\nLoading Σ>0 enumeration...")
    enum = pd.read_csv(DATA / "spaceship_enumeration.csv")
    enum_pos = enum[enum["Sigma"] > 0][["B_mask","S_mask"]].copy()
    df = df.merge(enum_pos.assign(sig=True), left_on=["B","S"], right_on=["B_mask","S_mask"], how="left").drop(columns=["B_mask","S_mask"])
    df["sig"] = df["sig"].fillna(False).astype(bool)

    # Save the per-rule table for later figure / discussion
    out_path = DATA / "symmetry_coords_full.parquet"
    df.to_parquet(out_path, index=False)
    print(f"Saved per-rule coordinate table to {out_path} ({len(df)} rows)")

    # Distributional summaries for the manuscript
    sig_pos = df[df["sig"]]; sig_neg = df[~df["sig"]]
    print()
    print("=== Δ distribution of Σ>0 (424 rules) vs Σ=0 (rest) ===")
    print(f"  Σ>0:  median Δ = {sig_pos['delta'].median():+.3f}, mean = {sig_pos['delta'].mean():+.3f}, "
          f"std = {sig_pos['delta'].std():.3f}  (n={len(sig_pos)})")
    print(f"  Σ=0:  median Δ = {sig_neg['delta'].median():+.3f}, mean = {sig_neg['delta'].mean():+.3f}, "
          f"std = {sig_neg['delta'].std():.3f}  (n={len(sig_neg)})")
    print(f"  Σ>0 Δ distribution:")
    for d in sorted(sig_pos["delta"].unique()):
        n = (sig_pos["delta"] == d).sum()
        bar = "#" * int(60 * n / len(sig_pos))
        print(f"    Δ={d:+.3f}  n={n:4d}  {bar}")

    print()
    print("=== Δ_edge analysis (the (T-odd, σ-odd) sector) ===")
    print(f"  All rules:  median Δ_edge = {df['delta_edge'].median():+.3f}, "
          f"mean = {df['delta_edge'].mean():+.3f}, std = {df['delta_edge'].std():.3f}")
    print(f"  Σ>0 rules: median Δ_edge = {sig_pos['delta_edge'].median():+.3f}, "
          f"mean = {sig_pos['delta_edge'].mean():+.3f}, std = {sig_pos['delta_edge'].std():.3f}")
    print(f"  Σ=0 rules: median Δ_edge = {sig_neg['delta_edge'].median():+.3f}, "
          f"mean = {sig_neg['delta_edge'].mean():+.3f}, std = {sig_neg['delta_edge'].std():.3f}")
    # Mann-Whitney U
    from scipy.stats import mannwhitneyu, pearsonr, spearmanr
    u, p = mannwhitneyu(sig_pos["delta_edge"], sig_neg["delta_edge"], alternative="two-sided")
    print(f"  Mann-Whitney U test Δ_edge | (Σ>0 vs Σ=0): U={u:.0f}, p={p:.3g}")

    print(f"\n  Δ_edge vs other coordinates (Pearson on all 262144 rules):")
    for c in ("lam","delta","mu_c","mu_p","mu"):
        r, p = pearsonr(df["delta_edge"], df[c])
        print(f"    r({c:6s}, Δ_edge) = {r:+.3f}  p={p:.3g}")

    # Independence check requested: Δ and μ
    print(f"\n  Cross-coordinate correlations (sanity / independence):")
    for c1, c2 in [("delta","mu_c"), ("delta","mu_p"), ("delta","mu"), ("lam","mu"), ("lam","delta")]:
        r, _ = pearsonr(df[c1], df[c2])
        print(f"    r({c1:6s}, {c2:6s}) = {r:+.3f}")

    # Show specific examples requested in the prompt
    print()
    print("=== Δ vs μ independence examples (from task prompt) ===")
    # B={3}, S={3}: rid = (8<<9)|8 = 4104
    # B={3}, S={5}: rid = (32<<9)|8 = 16392
    for label, B, S in [("B={3},S={3} (same Δ=0)", 8, 8),
                         ("B={3},S={5} (same Δ=0)", 8, 32),
                         ("Conway B={3}/S={2,3}", 8, 12),
                         ("Conway-swap B={2,3}/S={3}", 12, 8)]:
        h = df[(df["B"]==B) & (df["S"]==S)].iloc[0]
        print(f"  {label:36s}  λ={h['lam']:.3f}  Δ={h['delta']:+.3f}  μ_c={h['mu_c']:.3f}  μ={h['mu']:.3f}")
    print()
    print("DONE. All parities verified; safe to write the table.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
