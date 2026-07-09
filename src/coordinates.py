"""Rule-table coordinates for life-like cellular automata.

A life-like rule is fixed by two subsets B, S of {0,...,8} (birth and survival
condition on the Moore neighbour count). We represent B and S as 9-bit integer
masks (bit k is set iff k in the set).

Given (B_mask, S_mask), this module returns the paper's rule-table coordinates:

    lambda      = (|B| + |S|) / 18                                  # activity
    delta       = (|B| - |S|) / 9                                   # asymmetry
    mu_centre   = (1/256) sum_{n=0..8} C(8,n) * 1[n in B triangle S]
    mu_neigh    = (1/256) sum_{k=0..7} C(7,k) *
                      (1[k in B triangle (B-1)] + 1[k in S triangle (S-1)])
    mu          = (mu_centre + 8 * mu_neigh) / 9                     # sensitivity
    delta_edge  = (e(B) - e(S)) / 8    where e(M) = #{k: 1[k in M] != 1[k+1 in M]}

All formulas are closed-form functions of the rule table alone; nothing here
depends on any simulation. Arrays are vectorised over all 262144 rules.

Example:
    >>> from coordinates import coords, mask
    >>> B, S = mask({3}), mask({2,3})       # Conway B3/S23
    >>> lam, d, m, mc, mn, de = coords(B, S)
"""
from __future__ import annotations
from math import comb
from typing import Iterable
import numpy as np


# Pre-computed lookup tables indexed by the 9-bit mask (0 .. 511).
_C8 = [comb(8, n) for n in range(9)]
_C7 = [comb(7, k) for k in range(8)]

# popcount
POP = np.array([bin(m).count("1") for m in range(512)], dtype=np.int32)

# centre-bit sensitivity of a *symmetric-difference mask* m = B XOR S:
#     (1/256) * sum_{n in m} C(8, n)
CENTRE = np.array(
    [sum(_C8[n] for n in range(9) if (m >> n) & 1) / 256.0 for m in range(512)]
)

# neighbour-edge sensitivity of a single mask M (contribution before the
# 1/256 normalisation is per set): (1/256) sum_k C(7,k) 1[k in M XOR (M>>1)]
PERIPH = np.array(
    [
        sum(
            _C7[k] for k in range(8) if ((m >> k) & 1) != ((m >> (k + 1)) & 1)
        )
        / 256.0
        for m in range(512)
    ]
)

# edge count e(M) = # transitions in M as the count k rises 0->8.
EDGES = np.array(
    [
        sum(1 for k in range(8) if ((m >> k) & 1) != ((m >> (k + 1)) & 1))
        for m in range(512)
    ],
    dtype=np.int32,
)


def mask(s: Iterable[int]) -> int:
    """Turn a subset of {0..8} into a 9-bit mask."""
    m = 0
    for k in s:
        if not 0 <= int(k) <= 8:
            raise ValueError(f"neighbour count {k} out of range 0..8")
        m |= 1 << int(k)
    return m


def coords(B, S):
    """Coordinates for one or many rules.

    B, S may be Python ints or NumPy arrays of 9-bit masks.
    Returns (lambda, delta, mu, mu_centre, mu_neigh, delta_edge), each a scalar
    or array shaped like the inputs.
    """
    B = np.asarray(B)
    S = np.asarray(S)
    lam = (POP[B] + POP[S]) / 18.0
    delta = (POP[B] - POP[S]) / 9.0
    mu_centre = CENTRE[B ^ S]
    mu_neigh = PERIPH[B] + PERIPH[S]
    mu = (mu_centre + 8.0 * mu_neigh) / 9.0
    delta_edge = (EDGES[B] - EDGES[S]) / 8.0
    return lam, delta, mu, mu_centre, mu_neigh, delta_edge


def all_rules():
    """Return (B_mask, S_mask) arrays for all 262144 life-like rules.

    Rule id r encodes B in low 9 bits, S in high 9 bits: r = (S << 9) | B.
    """
    r = np.arange(262144, dtype=np.int32)
    return r & 0x1FF, (r >> 9) & 0x1FF


# ---- Symmetries used in Section 2.3 --------------------------------------
def _rev9(m: int) -> int:
    r = 0
    for k in range(9):
        if (m >> k) & 1:
            r |= 1 << (8 - k)
    return r


REV9 = np.array([_rev9(m) for m in range(512)], dtype=np.int32)


def tau(B, S):
    """(B, S) -> (S, B): birth/survival swap."""
    return S, B


def sigma(B, S):
    """(B, S) -> (~rev(S), ~rev(B)): bit-flip duality.

    For g = sigma(f), g(~x) = ~f(x) for every configuration x.
    """
    B = np.asarray(B)
    S = np.asarray(S)
    return (~REV9[S]) & 0x1FF, (~REV9[B]) & 0x1FF


if __name__ == "__main__":
    # Sanity: print the paper's anchor coordinates.
    anchors = {
        "Conway":      ({3}, {2, 3}),
        "HighLife":    ({3, 6}, {2, 3}),
        "2x2":         ({3, 6}, {1, 2, 5}),
        "Day&Night":   ({3, 6, 7, 8}, {3, 4, 6, 7, 8}),
        "Seeds":       ({2}, set()),
        "Replicator":  ({1, 3, 5, 7}, {1, 3, 5, 7}),
        "Maze":        ({3}, {1, 2, 3, 4, 5}),
    }
    print(f"{'rule':12s} {'lambda':>7s} {'delta':>7s} {'mu':>7s} "
          f"{'mu_c':>7s} {'d_edge':>7s}")
    for name, (Bset, Sset) in anchors.items():
        B, S = mask(Bset), mask(Sset)
        lam, d, m, mc, _, de = coords(B, S)
        print(f"{name:12s} {float(lam):7.3f} {float(d):+7.3f} {float(m):7.3f} "
              f"{float(mc):7.3f} {float(de):+7.3f}")
