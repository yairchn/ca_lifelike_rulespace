# Rule-table structure of life-like cellular automata

Reproduction code and derived data for the manuscript

> **Rule-table structure of life-like cellular automata: activity, asymmetry,
> sensitivity, and the propagation–storage factorisation.**
> Yair Cohen, 2026.

The paper studies the 262,144 two-dimensional outer-totalistic ("life-like")
cellular automata — the family that contains Conway's Game of Life — and asks
what small set of rule-table coordinates exposes the structure of behavioural
capability. It introduces two coordinates that complement Langton's λ:

- **δ = (|B| − |S|) / 9**, the birth–survival asymmetry, and
- **μ**, the average Boolean-derivative sensitivity of Vichniac and Binder,
  which splits in closed form into a centre-bit term **μ_centre** (zero
  exactly on the fully totalistic B = S submanifold) and a neighbour term.

Combined with an exhaustive enumeration of every rule's compact (3×3, T ≤ 16)
gliders, still lifes and oscillators, the coordinates yield a hierarchical
B / S factorisation and the compact-glider exclusion on B = S.

The manuscript itself lives in a separate working tree; this repository ships
only the reproduction code and the derived data needed to rebuild every figure.

## Repository layout

```
.
├── src/              Rule-table coordinate code
│   ├── coordinates.py             λ, δ, μ, μ_centre, μ_neigh, δ_edge, τ, σ
│   └── build_symmetry_coords.py   Verify parities on all 262 144 rules and
│                                  write data/symmetry_coords_full.parquet
├── figures/          One script per manuscript figure
│   ├── fig1_binder_planes.py         → binder_lambda_mu.png
│   ├── fig2_sigma_characterization.py→ sigma_characterization.png
│   ├── fig3_delta_histograms.py      → Fig_delta_histograms.png
│   ├── fig4_factorization.py         → factorization.png
│   └── fig5_joint_plane.py           → joint_plane.png
├── outputs/          Figure PNGs land here after running the scripts
└── data/             Derived data required by the figure scripts
    ├── spaceship_enumeration.csv     φ (3×3 compact-spaceship count) per rule
    ├── spaceship_enumeration_4x4_full.csv  4×4 spaceship enumeration
    │                                  (used for the k≥1 propagation-gate
    │                                  containment 26 ⊂ 64 in §3.4)
    ├── spaceships_detail.jsonl       One line per found (rule, seed, period, v)
    ├── still_life_enumeration.parquet  ψ per rule
    ├── oscillator_enumeration.csv    ω per rule
    ├── composite_scores.csv          per-rule combined table (names, G*_alt, …)
    ├── silent38_summary.csv          33 robustly-silent + 5 rare-basin split
    │                                  used by fig5 (§3.3 stress test)
    ├── manifest_mu_full.parquet      Pre-computed μ table (used for cross-check)
    └── symmetry_coords_full.parquet  λ, δ, μ, μ_c, μ_n, δ_edge per rule
```

## Install

```bash
git clone <this-repo> ca_lifelike_rulespace
cd ca_lifelike_rulespace
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Tested with Python 3.11 and the versions pinned in `requirements.txt`.

## Reproduce the figures

Every figure in the manuscript can be rebuilt from the shipped `data/`:

```bash
python figures/fig1_binder_planes.py
python figures/fig2_sigma_characterization.py
python figures/fig3_delta_histograms.py
python figures/fig4_factorization.py
python figures/fig5_joint_plane.py
```

Each writes its PNG into `outputs/`.

The `symmetry_coords_full.parquet` table used by fig3 is shipped but can be
regenerated (with a parity self-check on all 262 144 rules) via

```bash
python src/build_symmetry_coords.py
```

## Rule-table coordinates in your own code

The `coordinates` module is a small, self-contained implementation of the
paper's coordinates:

```python
from src.coordinates import mask, coords

B, S = mask({3}), mask({2, 3})                    # Conway B3/S23
lam, delta, mu, mu_centre, mu_neigh, delta_edge = coords(B, S)
# lam = 0.167, delta = -0.111, mu = 0.352, mu_centre = 0.109, ...
```

It vectorises over NumPy arrays, so you can compute coordinates for all
262 144 rules in one call:

```python
from src.coordinates import all_rules, coords
B, S = all_rules()
lam, delta, mu, mu_centre, mu_neigh, delta_edge = coords(B, S)
```

> **Note.** The paper's analysis uses only `lam`, `delta`, `mu`, and
> `mu_centre`. The auxiliary coordinates `mu_neigh` (neighbour-bit
> sensitivity) and `delta_edge` (edge-imbalance, τ-odd/σ-odd) are computed
> by `coordinates.py` for provenance and reproducibility, but are not
> carried through the manuscript's arguments.

## Data provenance

The `data/spaceship_enumeration.csv`, `still_life_enumeration.parquet`, and
`oscillator_enumeration.csv` files are the outputs of exhaustive enumerations
of every 3×3 D₄-orbit seed under each of the 262 144 rules, run to T = 16.
See §2.5 of the manuscript for the enumeration protocol. The
`composite_scores.csv` merges these with the dynamical probe G* (§2.6).
`manifest_mu_full.parquet` and `symmetry_coords_full.parquet` are closed-form
functions of (B, S) alone; both can be regenerated from `src/coordinates.py`
and `src/build_symmetry_coords.py`.

## Citing

If you use this code or data, please cite the manuscript.

## Licence

MIT — see [LICENSE](LICENSE).
