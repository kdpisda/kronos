---
title: Symmetry, Space Groups, and IBZ k-Point Folding
description: How KRONOS uses spglib for space-group detection, irreducible Brillouin-zone k-point reduction, time-reversal folding, density and force symmetrization in plane-wave DFT.
keywords:
  - DFT symmetry
  - space group DFT
  - spglib
  - irreducible Brillouin zone
  - IBZ k-point reduction
  - time-reversal symmetry DFT
  - Monkhorst-Pack folding
  - density symmetrization
  - force symmetrization
  - plane-wave DFT symmetry
slug: /physics/symmetry-spglib
sidebar_position: 13
---

# Symmetry, Space Groups, and IBZ k-Point Folding

Exploiting crystal symmetry is the single largest source of speedup in plane-wave DFT calculations of solids. A face-centered-cubic crystal has 48 point-group operations; using only the irreducible Brillouin zone (IBZ) instead of the full BZ cuts the eigenvalue work by ~48× without changing a single output number. KRONOS detects the space group automatically via spglib and applies symmetry consistently throughout the SCF loop: k-point reduction, density symmetrization, and force/stress symmetrization. This page covers how each piece works and why it matters for accuracy.

## Why symmetry matters in DFT

Three places where symmetry enters a plane-wave DFT calculation:

1. **K-point sampling** — the BZ integral $\int_\mathrm{BZ} d\mathbf{k}\,\ldots$ is invariant under point-group rotations. Sampling only the IBZ and weighting each point by its multiplicity gives the same integral with fewer eigensolves.
2. **Density symmetrization** — the converged electron density must obey the space-group symmetry of the crystal. Numerical noise (finite k-mesh, finite SCF tolerance) breaks this; symmetrization at each SCF step removes the noise and improves convergence stability.
3. **Forces and stress** — these must transform as covariant tensors under symmetry. Symmetrizing them eliminates floating-point noise and ensures geometry-optimization paths preserve symmetry.

KRONOS handles all three via [spglib](https://spglib.github.io/spglib/), the standard library for crystallographic symmetry detection.

## Space-group detection

Given a crystal (lattice vectors $\mathbf{a}_1, \mathbf{a}_2, \mathbf{a}_3$ plus atomic positions $\boldsymbol\tau_a$ and species), spglib returns:

- **Space-group number** (1–230 in international tables)
- **Hall symbol** and Hermann-Mauguin notation
- **Set of symmetry operations** $\{(\mathbf{R}, \mathbf{t})\}$ — point-group rotation $\mathbf{R}$ (a $3\times 3$ matrix in fractional coordinates) plus fractional translation $\mathbf{t}$
- **Wyckoff positions** of each atom

KRONOS calls spglib at startup once the crystal is built. If spglib is not linked (no `KRONOS_HAS_SPGLIB`), the calculation falls back to point group 1 (identity only) — every k-point is "irreducible" and no symmetrization happens. This is correct but slower.

## Monkhorst-Pack k-point grids

Without symmetry, a Monkhorst-Pack grid $N_1 \times N_2 \times N_3$ generates $N_1 N_2 N_3$ k-points, each with weight $w_\mathbf{k} = 1/N_1 N_2 N_3$. The Brillouin-zone integral becomes a sum:

$$\frac{1}{\Omega_\mathrm{BZ}} \int_\mathrm{BZ} f(\mathbf{k})\, d\mathbf{k} \to \sum_\mathbf{k} w_\mathbf{k}\, f(\mathbf{k})$$

A shifted grid (offset by $\tfrac{1}{2}$ in each direction) often gives faster convergence by avoiding high-symmetry points where degeneracies create discontinuities in the integrand.

## IBZ folding via symmetry

K-points related by point-group operations $\mathbf{R}$ give the same wavefunction up to a unitary rotation, so the eigenvalues are identical. KRONOS folds the full MP grid into the IBZ by:

1. For each k-point $\mathbf{k}$, compute its orbit under all $\mathbf{R}$ in the point group: $\{\mathbf{R}\mathbf{k}\}$.
2. The orbit representative (the unique k-point in the IBZ) absorbs the weights of all its images.
3. The total weight of the IBZ rep is $w = m / N_1 N_2 N_3$ where $m$ is the orbit size (the multiplicity).

Time-reversal symmetry (always present in spin-unpolarized calculations) effectively doubles the symmetry group by adding $\mathbf{k} \to -\mathbf{k}$, halving the IBZ further. In spin-polarized calculations without spin-orbit coupling, time-reversal still applies; with spin-orbit it does not.

A face-centered-cubic crystal with a $4\times 4\times 4$ shifted grid has 64 full-BZ points but only 8 in the IBZ — a 64/8 = 8× speedup for the eigenvalue work. For Si diamond ($O_h$ symmetry, 48 operations) on a $4\times 4\times 4$ Monkhorst-Pack grid, the count drops to 10 IBZ points (some special points have smaller orbits than 48).

## Density symmetrization in G-space

The electron density $n(\mathbf{r})$ should be invariant under every space-group operation $(\mathbf{R}, \mathbf{t})$:

$$n(\mathbf{R}\mathbf{r} + \mathbf{t}) = n(\mathbf{r})$$

In G-space, this translates to a constraint relating Fourier components at symmetry-related G-vectors:

$$n(\mathbf{R}\mathbf{G}) = e^{i\mathbf{R}\mathbf{G}\cdot\mathbf{t}}\, n(\mathbf{G})$$

KRONOS symmetrizes the density at each SCF step by averaging over the point group:

$$n^\mathrm{sym}(\mathbf{G}) = \frac{1}{|G|} \sum_{(\mathbf{R}, \mathbf{t}) \in G} e^{-i\mathbf{R}\mathbf{G}\cdot\mathbf{t}}\, n(\mathbf{R}\mathbf{G})$$

This step is critical for multi-k-point convergence: without it, the truncated k-mesh introduces tiny asymmetries that the Hartree response amplifies into SCF instability. The classic case is FCC Si with a $4\times 4\times 4$ shifted mesh — KRONOS without density symmetrization was off by ~4 meV/atom vs QE; with it, the error is 0.15 meV/atom (a 28× improvement).

**Important exception:** for Gamma-only calculations (1×1×1 k-grid), density symmetrization is skipped. A Gamma-only density is already exactly symmetric, and applying the symmetrization can introduce DIIS instability from numerical noise in the averaging.

## Force and stress symmetrization

Forces transform covariantly: if atoms $a$ and $a'$ are related by symmetry operation $\mathbf{R}$, then $\mathbf{F}_{a'} = \mathbf{R}\,\mathbf{F}_a$. KRONOS symmetrizes by mapping each atom under every operation and averaging the rotated forces:

$$\mathbf{F}_a^\mathrm{sym} = \frac{1}{|G_a|} \sum_{\mathbf{R} \in G_a} \mathbf{R}^{-1}\, \mathbf{F}_{\sigma_\mathbf{R}(a)}$$

where $G_a$ is the site-symmetry group of atom $a$ and $\sigma_\mathbf{R}$ is the permutation of atoms induced by $\mathbf{R}$. The same idea applied to the stress tensor produces a tensor that respects the full point-group symmetry of the cell.

Force symmetrization automatically enforces Newton's third law (translation invariance): $\sum_a \mathbf{F}_a^\mathrm{sym} = 0$. KRONOS asserts this in unit tests with a tolerance of $10^{-12}$ Ry/bohr.

## Implementation in KRONOS

The symmetry pipeline lives in `src/io/symmetry.cpp`:

- `SymmetryAnalyzer::detect(crystal)` — calls spglib and stores the operation list
- `SymmetryAnalyzer::reduce_kpoints(grid)` — folds the MP grid into the IBZ with weights
- `SymmetryAnalyzer::symmetrize_density(n_g, gvecs)` — G-space density symmetrization
- `SymmetryAnalyzer::symmetrize_forces(forces, atoms)` — covariant force symmetrization
- `SymmetryAnalyzer::symmetrize_stress(sigma)` — 3×3 tensor symmetrization

When `KRONOS_HAS_SPGLIB` is not defined (spglib unavailable at build time), all these methods become no-ops, and the calculation falls back to no symmetry — the result is still correct, just slower.

## References

- Togo, A. & Tanaka, I. "Spglib: a software library for crystal symmetry search", *arXiv:1808.01590* (2018). Spglib documentation: <https://spglib.github.io/spglib/>
- Monkhorst, H. J. & Pack, J. D. "Special points for Brillouin-zone integrations", *Phys. Rev. B* **13**, 5188 (1976).
- Ramírez, R. & Böhm, M. C. "Simple geometric generation of special points in Brillouin-zone integrations: Two-dimensional Bravais lattices", *Int. J. Quantum Chem.* **30**, 391 (1986).
- Martin, R. M. *Electronic Structure*, Ch. 4 — symmetry in solids.
- International Tables for Crystallography, Vol. A (space-group descriptions).
