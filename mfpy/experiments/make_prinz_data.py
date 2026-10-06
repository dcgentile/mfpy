"""
Regenerate ``data/prinz/prinz.txt``.

``Notebooks/Prinz/prinz-figs.ipynb`` loads ``data/prinz/prinz.txt``, but that file is not
in the repository (it is not in ``.gitignore`` either -- it appears simply never to have
been committed). This script regenerates a trajectory from the same source the notebook's
imports imply: ``deeptime``'s Prinz quadruple-well potential
:footcite:`prinz2011markov`,

    V(x) = 4 (x^8 + 0.8 e^{-80 x^2} + 0.2 e^{-80 (x - 0.5)^2} + 0.5 e^{-40 (x + 0.5)^2}),

integrated with Euler-Maruyama at the package defaults (h = 1e-5, 500 steps between
recorded samples, kT = 1, mass 1, damping 1), which are the settings the notebook uses
implicitly by calling the constructor with no arguments.

**This is a regeneration, not the original file.** The seed below fixes the trajectory so
results are reproducible from here on, but numbers computed on this trajectory will not
match any that were computed on the original. Anything quoted from the old notebook
should be recomputed rather than compared.

Usage
-----
    python mfpy/experiments/make_prinz_data.py [--length 10000] [--seed 42]
"""

from __future__ import annotations

import argparse
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(REPO_ROOT, "data", "prinz", "prinz.txt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--length", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--x0", type=float, default=-0.5)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    from deeptime.data import prinz_potential

    system = prinz_potential()
    print("minima:", system.minima)
    traj = system.trajectory(x0=[[args.x0]], length=args.length,
                             seed=args.seed).astype(float).ravel()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savetxt(args.out, traj, fmt="%.10f")

    edges = np.concatenate([[-np.inf],
                            (system.minima[1:] + system.minima[:-1]) / 2,
                            [np.inf]])
    occ = np.histogram(traj, bins=edges)[0]
    print(f"wrote {args.out}: T={traj.shape[0]}, "
          f"range [{traj.min():.3f}, {traj.max():.3f}]")
    print("well occupancy:", dict(zip(np.round(system.minima, 3), occ)))


if __name__ == "__main__":
    main()
