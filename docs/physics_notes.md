# KRONOS Physics Notes

## What KRONOS Solves

KRONOS solves the **Kohn-Sham equations** of Density Functional Theory (DFT) for periodic crystalline systems. DFT maps the interacting many-electron problem onto a set of independent single-particle equations that yield the exact ground-state electron density (in principle).

### The Kohn-Sham Equations

For each Kohn-Sham orbital ψ_nk:

```
[-∇²/2 + V_eff(r)] ψ_nk(r) = ε_nk ψ_nk(r)
```

where the effective potential is:

```
V_eff(r) = V_H(r) + V_xc(r) + V_ext(r)
```

- **V_H(r)**: Hartree (classical electron-electron Coulomb) potential
- **V_xc(r)**: Exchange-correlation potential (approximated)
- **V_ext(r)**: External potential (ions, via pseudopotentials)

The density is constructed from the orbitals:

```
ρ(r) = Σ_{nk} f_nk |ψ_nk(r)|²
```

Since V_eff depends on ρ, which depends on ψ, which depends on V_eff, the equations must be solved **self-consistently** (SCF loop).

---

## Plane-Wave Formalism

### Why Plane Waves?

In periodic crystals, Bloch's theorem states that wavefunctions can be written:

```
ψ_nk(r) = e^{ik·r} u_nk(r)
```

where u_nk(r) has the periodicity of the lattice. Expanding u_nk in a Fourier series:

```
ψ_nk(r) = Σ_G c_{nk}(G) e^{i(k+G)·r}
```

This is the plane-wave expansion. The coefficients c_{nk}(G) are what KRONOS stores and manipulates.

### The Cutoff

The expansion is truncated at a finite kinetic energy cutoff:

```
|k + G|² / 2 ≤ E_cut    (ecutwfc in Ry)
```

This determines the basis size (number of plane waves). Higher cutoff = more plane waves = more accurate but slower.

### Why Plane Waves are Efficient

1. **Kinetic energy**: Diagonal in G-space: T_G = |k+G|²/2
2. **Local potentials**: Diagonal in r-space: V(r)·ψ(r)
3. **Switching via FFT**: O(N log N) transforms between representations
4. **Systematic convergence**: Single parameter (ecutwfc) controls accuracy
5. **No basis set superposition error** (unlike Gaussian basis sets)

---

## Pseudopotential Theory

### The Problem

Core electrons (1s, 2s, 2p, ...) are tightly bound and don't participate in bonding but create rapid oscillations in the valence wavefunctions near the nucleus. These oscillations require enormous plane-wave cutoffs to resolve.

### The Solution: Pseudopotentials

Replace the true ionic potential + core electrons with a smooth **pseudopotential** that:
1. Reproduces the correct valence eigenvalues
2. Matches the true wavefunction outside a cutoff radius r_c
3. Produces smooth pseudo-wavefunctions inside r_c

### Norm-Conserving Pseudopotentials

KRONOS v0.1 uses norm-conserving (NC) pseudopotentials, which satisfy:

```
∫₀^{r_c} |φ̃_l(r)|² r² dr = ∫₀^{r_c} |φ_l(r)|² r² dr
```

This ensures correct charge transfer between atoms.

### Kleinman-Bylander Form

The semilocal pseudopotential V_PS = V_loc(r) + Σ_l |l⟩ δV_l ⟨l| is rewritten in separable (KB) form:

```
V_NL = Σ_l |β_l⟩ D_l ⟨β_l|
```

where β_l are projector functions derived from δV_l · φ_l. This is much more efficient: projections ⟨β|ψ⟩ are computed once, then V_NL|ψ⟩ is a sum of projectors weighted by D_l.

### UPF Format

KRONOS reads the Unified Pseudopotential Format (UPF v2), which contains:
- Radial grid r(i) and integration weights rab(i)
- Local potential V_loc(r) on the radial grid
- Beta projectors β_l(r) for each angular momentum channel
- D_ij coupling matrix
- Atomic wavefunctions and density for initialization

---

## Exchange-Correlation Functionals

The XC functional is the only approximation in Kohn-Sham DFT. KRONOS supports:

### LDA (Local Density Approximation)

```
E_xc[ρ] = ∫ ρ(r) ε_xc(ρ(r)) dr
```

The XC energy density depends only on the local density. Implementations:
- **LDA_PZ**: Perdew-Zunger (1981) parametrization of Ceperley-Alder QMC data
- **LDA_PW**: Perdew-Wang (1992) parametrization

### GGA (Generalized Gradient Approximation)

```
E_xc[ρ] = ∫ ρ(r) ε_xc(ρ(r), |∇ρ(r)|²) dr
```

Also depends on the density gradient. Implementations:
- **PBE**: Perdew-Burke-Ernzerhof (1996) — the most widely used GGA
- **PBEsol**: PBE revised for solids (better lattice constants)

### LDA vs GGA

| Property | LDA | GGA |
|----------|-----|-----|
| Binding energies | Overbinds ~1 eV | Better, ~0.3 eV |
| Lattice constants | Underestimates ~1% | Better for PBEsol |
| Band gaps | Underestimates (both) | Underestimates (both) |
| Computational cost | Baseline | ~10% more (gradient) |

---

## Ewald Summation

### The Problem

The electrostatic energy of periodic point charges:

```
E_ion = (1/2) Σ_{i≠j} Σ_T Z_i Z_j / |r_i - r_j + T|
```

converges extremely slowly (conditionally convergent).

### Ewald's Solution

Split the 1/r interaction into a short-range part (converges fast in real space) and a smooth part (converges fast in reciprocal space):

```
1/r = erfc(ηr)/r + erf(ηr)/r
```

The total energy has four terms:
1. **Real-space sum**: Σ'_{T} Z_i Z_j erfc(η|r_ij+T|)/|r_ij+T| — short-ranged, few terms needed
2. **Reciprocal-space sum**: (4π/Ω) Σ_{G≠0} |S(G)|² exp(-G²/4η²)/G² — smooth, few G-vectors
3. **Self-energy correction**: -η/√π Σ_i Z_i² — removes self-interaction
4. **Charged system correction**: for non-neutral cells

### Madelung Constants

For simple ionic crystals, the Ewald energy reduces to:

```
E = -α_M N Z₊Z₋ / a_nn
```

where α_M is the Madelung constant (geometry-dependent):
- NaCl: α = 1.74756
- CsCl: α = 1.76267
- ZnS (zincblende): α = 1.63806

KRONOS's Ewald implementation is validated against these known values.

---

## Hellmann-Feynman Forces

The force on atom I is:

```
F_I = -dE/dR_I = -⟨ψ| dH/dR_I |ψ⟩
```

By the Hellmann-Feynman theorem, at the SCF minimum, the force equals the expectation value of the potential derivative. KRONOS computes three contributions:

1. **Ewald forces**: dE_ion/dR_I (from the Ewald real and reciprocal sums)
2. **Local PP forces**: d/dR_I ∫ V_loc(r-R_I) ρ(r) dr (structure factor derivative)
3. **Nonlocal PP forces**: d/dR_I Σ_n f_n ⟨ψ_n|V_NL|ψ_n⟩ (projector derivative)

Forces are used for:
- Verifying equilibrium (forces = 0 at symmetric positions)
- Geometry optimization (BFGS minimization)
- Molecular dynamics (not yet implemented)

---

## What to Converge and Why

### ecutwfc (Plane-Wave Cutoff)

Controls basis set completeness. Must converge:
- Total energy: increase until ΔE < 1 meV/atom
- Forces: increase until ΔF < 1 meV/Å
- Stress: increase until Δσ < 0.1 kbar

The variational principle guarantees E(higher cutoff) ≤ E(lower cutoff).

### K-point Grid

Controls Brillouin zone sampling. Must converge:
- Total energy to < 1 meV/atom
- Metals need denser grids than insulators
- Symmetry reduces the number of irreducible k-points

### Smearing Width (degauss)

For metals only. Controls the occupation function width:
- Too large: artificial electronic temperature, wrong energy
- Too small: poor SCF convergence
- Typical: 0.01-0.02 Ry for Marzari-Vanderbilt smearing
- Extrapolate to degauss→0 for publication-quality results

### SCF Convergence Threshold

Controls when the self-consistency loop stops:
- energy_threshold: |E_new - E_old| < threshold
- density_threshold: ||ρ_new - ρ_old|| < threshold
- For forces: need tight SCF (1e-8 Ry energy, 1e-9 density)
- For total energy only: 1e-6 Ry usually sufficient
