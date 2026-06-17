---
title: The Kohn-Sham Equations
description: Hohenberg-Kohn theorems, the Kohn-Sham ansatz, self-consistent equations, and the DFT total energy functional in the plane-wave KRONOS implementation.
keywords:
  - Kohn-Sham equations
  - Hohenberg-Kohn theorem
  - density functional theory
  - DFT
  - self-consistent field
  - SCF
  - exchange-correlation potential
  - DFT total energy
  - plane-wave DFT
  - Kohn-Sham ansatz
  - electronic structure
  - many-electron problem
  - band structure DFT
  - Janak theorem
slug: /physics/kohn-sham-equations
sidebar_position: 4
---

# The Kohn-Sham Equations

Density functional theory (DFT) reduces the exponentially complex quantum many-body problem to a set of effective single-particle equations that are solved self-consistently. The Kohn-Sham (KS) formulation, published in 1965, is the practical engine behind virtually every modern electronic structure calculation — including every SCF iteration in KRONOS. This page derives the KS equations from first principles, explains what they mean, and maps them to KRONOS's implementation.

All equations use **Rydberg atomic units** ($\hbar = 1$, $m_e = 1/2$, $e^2 = 2$, $a_0 = 1$) throughout, matching the KRONOS source. The key consequences: the kinetic operator is $-\nabla^2$ (no factor of $1/2$), the bare Coulomb potential is $-2Z/r$, and Hartree energy prefactors carry $8\pi$ instead of $4\pi$.

## The many-electron problem

A system of $N$ electrons moving under the external potential of $M$ nuclei is described by the full many-body Hamiltonian

$$\hat{H} = \sum_{i=1}^N \left[ -\nabla_i^2 + v_\mathrm{ext}(\mathbf{r}_i) \right] + \sum_{i < j} \frac{2}{|\mathbf{r}_i - \mathbf{r}_j|}$$

where $v_\mathrm{ext}(\mathbf{r}) = -\sum_a 2Z_a / |\mathbf{r} - \boldsymbol{\tau}_a|$ is the electron-nuclear attraction (Rydberg Coulomb, $2/r$). The exact ground-state wavefunction $\Psi(\mathbf{r}_1, \sigma_1, \ldots, \mathbf{r}_N, \sigma_N)$ lives in a $3N$-dimensional Hilbert space and carries all physical information.

The cost is exponential: to represent $\Psi$ on a real-space grid with $M_\mathrm{grid}$ points per dimension per electron requires $M_\mathrm{grid}^{3N}$ complex numbers. For Si with 8 valence electrons and a modest grid of $20^3$ points per electron, that is $20^{24} \approx 10^{31}$ numbers — completely intractable. The electron-electron Coulomb repulsion $\sum_{i<j} 2/|\mathbf{r}_i - \mathbf{r}_j|$ is the term that couples all electrons together and prevents factoring the problem into independent single-particle equations. DFT dissolves this wall by reformulating everything in terms of the electron density $n(\mathbf{r})$, a function of only three spatial coordinates.

## Hohenberg-Kohn theorems

Hohenberg and Kohn (1964) proved two theorems that provide the formal foundation for DFT.

### Theorem 1: density determines potential

For a non-degenerate ground state, the external potential $v_\mathrm{ext}(\mathbf{r})$ is uniquely determined by the ground-state electron density $n(\mathbf{r})$, up to an additive constant.

The proof is by contradiction. Suppose two potentials $v_\mathrm{ext}$ and $v_\mathrm{ext}'$ (differing by more than a constant) yield the same ground-state density $n(\mathbf{r})$. They have different ground states $\Psi$ and $\Psi'$ with energies $E$ and $E'$. Applying the variational principle twice — using $\Psi'$ as a trial state for $H$, and $\Psi$ as a trial state for $H'$ — and adding the two inequalities yields a strict contradiction $E + E' < E + E'$. Therefore no two distinct potentials can share the same ground-state density.

The implication is deep: since $v_\mathrm{ext}$ fixes $H$, which fixes $\Psi$ (and all observables), **the ground-state density $n(\mathbf{r})$ alone contains all ground-state information**. Every observable is a functional of $n$.

### Theorem 2: variational principle

There exists a universal functional $F[n]$, independent of $v_\mathrm{ext}$, such that the total-energy functional

$$E_{v_\mathrm{ext}}[n] = F[n] + \int v_\mathrm{ext}(\mathbf{r}) \, n(\mathbf{r}) \, d\mathbf{r}$$

is minimized by the true ground-state density $n_0(\mathbf{r})$:

$$E_0 = \min_n E_{v_\mathrm{ext}}[n] = E_{v_\mathrm{ext}}[n_0].$$

$F[n] = T[n] + E_{ee}[n]$ contains the universal kinetic and electron-electron interaction energies. The second term, $\int v_\mathrm{ext} \, n$, is the only part that depends on the particular system.

**These are existence theorems, not constructions.** They tell us that the exact $F[n]$ exists, but they give no recipe for computing it. In particular, $T[n]$ — the kinetic energy as a functional of density alone — is not known exactly for an interacting system. This gap is precisely what the Kohn-Sham ansatz fills.

## The Kohn-Sham ansatz

Kohn and Sham's key insight is to sidestep the unknown $T[n]$ by introducing a *fictitious non-interacting reference system*: a set of $N$ independent electrons moving in an effective local potential $v_\mathrm{eff}(\mathbf{r})$ chosen so that the non-interacting ground-state density exactly equals the density of the real interacting system.

Because the reference electrons are non-interacting, their ground state is a Slater determinant built from $N$ single-particle orbitals $\{\psi_i(\mathbf{r})\}$ — the Kohn-Sham orbitals. The density is

$$n(\mathbf{r}) = \sum_i f_i \, |\psi_i(\mathbf{r})|^2$$

where $f_i$ are occupation numbers (0 or 1 for insulators; fractional for metals with smearing). The kinetic energy of the non-interacting reference, $T_s[n] = \sum_i f_i \langle \psi_i | {-\nabla^2} | \psi_i \rangle$, is computable exactly from the orbitals. The difference between the true kinetic energy $T[n]$ and $T_s[n]$ is absorbed into the exchange-correlation functional $E_\mathrm{xc}[n]$, along with the non-classical part of the electron-electron interaction.

## The Kohn-Sham equations

The stationary condition $\delta E[n] / \delta n = 0$ under the constraint $\int n \, d\mathbf{r} = N$ yields the Kohn-Sham eigenvalue equations. In Rydberg atomic units:

$$\boxed{\left[ -\nabla^2 + v_\mathrm{eff}(\mathbf{r}) \right] \psi_i(\mathbf{r}) = \varepsilon_i \, \psi_i(\mathbf{r})}$$

The effective potential is

$$v_\mathrm{eff}(\mathbf{r}) = v_\mathrm{ext}(\mathbf{r}) + v_\mathrm{H}[n](\mathbf{r}) + v_\mathrm{xc}[n](\mathbf{r})$$

where the three contributions are:

- **External potential** $v_\mathrm{ext}(\mathbf{r})$: electron-nuclear attraction plus any applied field. In KRONOS, the nuclear contribution is handled through pseudopotentials (local + nonlocal separable projectors in the Kleinman-Bylander form), not the bare $-2Z/r$ singularity.

- **Hartree potential** $v_\mathrm{H}(\mathbf{r})$: classical electrostatic potential of the electron charge cloud,
  $$v_\mathrm{H}(\mathbf{r}) = \int \frac{2 \, n(\mathbf{r}')}{|\mathbf{r} - \mathbf{r}'|} \, d\mathbf{r}'.$$
  In Rydberg units the Coulomb kernel is $2/r$, so in reciprocal space: $v_\mathrm{H}(\mathbf{G}) = 8\pi \, n(\mathbf{G}) / |\mathbf{G}|^2$ (the $8\pi$ Hartree prefactor instead of $4\pi$).

- **Exchange-correlation potential** $v_\mathrm{xc}(\mathbf{r})$: functional derivative of $E_\mathrm{xc}$ with respect to density,
  $$v_\mathrm{xc}(\mathbf{r}) = \frac{\delta E_\mathrm{xc}[n]}{\delta n(\mathbf{r})}.$$

These are $N$ coupled differential equations of single-particle form — structurally identical to the Schrödinger equation for a particle in an external potential. The crucial distinction from the real problem is that the coupling between electrons has been replaced by the effective potential $v_\mathrm{eff}$, which every electron feels identically.

## Self-consistency

The KS equations are inherently nonlinear: $v_\mathrm{eff}$ depends on $n(\mathbf{r})$, which depends on the orbitals $\{\psi_i\}$, which depend on $v_\mathrm{eff}$. They must be solved iteratively:

1. **Initial guess**: construct $n^{(0)}(\mathbf{r})$ from superposition of atomic densities.
2. **Build potential**: compute $v_\mathrm{H}[n]$ (Poisson solve in G-space), evaluate $v_\mathrm{xc}[n]$ via libxc, add $v_\mathrm{ext}$.
3. **Diagonalize**: solve $[-\nabla^2 + v_\mathrm{eff}]\psi_i = \varepsilon_i \psi_i$ for each k-point.
4. **New density**: form $n^{(\mathrm{new})}(\mathbf{r}) = \sum_i f_i |\psi_i(\mathbf{r})|^2$.
5. **Mixing**: blend $n^{(\mathrm{new})}$ with $n^{(\mathrm{old})}$ using Pulay/DIIS density mixing (history depth 8) to accelerate convergence and suppress charge sloshing.
6. **Convergence check**: if $\|n^{(\mathrm{new})} - n^{(\mathrm{old})}\|$ and $|E^{(\mathrm{new})} - E^{(\mathrm{old})}|$ are both below tolerance, stop. Otherwise return to step 2.

This self-consistent field (SCF) cycle is the central loop of every DFT code. For the operational view of how KRONOS implements it — timings, convergence thresholds, abort conditions — see [SCF Flowchart](/docs/architecture/scf-flowchart).

## The DFT total energy

Once self-consistency is reached, the total energy is assembled from its components. In Rydberg units:

$$E_\mathrm{tot} = T_s[n] + E_\mathrm{ext}[n] + E_\mathrm{H}[n] + E_\mathrm{xc}[n] + E_\mathrm{ion-ion}$$

Each term has a precise definition:

- **Non-interacting kinetic energy** $T_s[n]$: computed from the KS orbitals,
  $$T_s[n] = \sum_i f_i \langle \psi_i | {-\nabla^2} | \psi_i \rangle = \sum_i f_i \sum_\mathbf{G} |\mathbf{k}+\mathbf{G}|^2 |c_i(\mathbf{G})|^2.$$
  In G-space this is a simple sum over plane-wave coefficients — no FFT required.

- **External energy** $E_\mathrm{ext}[n] = \int v_\mathrm{ext}(\mathbf{r}) \, n(\mathbf{r}) \, d\mathbf{r}$: electron-nuclear interaction, carried through pseudopotential matrix elements.

- **Hartree energy** $E_\mathrm{H}[n]$: classical electrostatic self-energy,
  $$E_\mathrm{H}[n] = \frac{1}{2}\int\!\int \frac{2 \, n(\mathbf{r}) \, n(\mathbf{r}')}{|\mathbf{r}-\mathbf{r}'|} \, d\mathbf{r} \, d\mathbf{r}' = \frac{\Omega}{2} \sum_{\mathbf{G} \neq 0} v_\mathrm{H}(\mathbf{G}) \, n^*(\mathbf{G}).$$
  The $1/2$ avoids double-counting (each electron pair counted once). In Rydberg units $v_\mathrm{H}(\mathbf{G}) = 8\pi n(\mathbf{G})/|\mathbf{G}|^2$, so $E_\mathrm{H}$ carries the factor $4\pi$.

- **Exchange-correlation energy** $E_\mathrm{xc}[n]$: the only term that is not known exactly. LDA approximates it as $\int \varepsilon_\mathrm{xc}(n(\mathbf{r})) \, n(\mathbf{r}) \, d\mathbf{r}$; GGA adds dependence on $|\nabla n|$. KRONOS delegates to libxc for all but its built-in LDA fallback.

- **Ion-ion interaction** $E_\mathrm{ion-ion}$: classical Coulomb repulsion between nuclei. This diverges for a periodic system and requires the Ewald summation technique (a separate topic).

In practice KRONOS computes $E_\mathrm{tot}$ from the band energy sum via double-counting corrections (see `scf.cpp`):

$$E_\mathrm{tot} = E_\mathrm{band} - E_\mathrm{H} + E_\mathrm{xc} - \int v_\mathrm{xc}(\mathbf{r}) \, n(\mathbf{r}) \, d\mathbf{r} + E_\mathrm{Ewald}$$

where $E_\mathrm{band} = \sum_i f_i \varepsilon_i$ is the sum of KS eigenvalues. The subtracted terms correct for the fact that $E_\mathrm{band}$ double-counts $E_\mathrm{H}$ and $E_\mathrm{xc}$.

For metallic systems with fractional occupations, a smearing entropy term $-T S = -\sigma \cdot s$ is included (where $\sigma$ is the smearing width and $s$ is the dimensionless entropy), and the quantity minimized is the free energy $F = E_\mathrm{tot} - TS$.

## Kohn-Sham eigenvalues and band structure

The KS eigenvalues $\varepsilon_i$ are often plotted as band structures and compared to photoemission experiments, but their physical interpretation requires care.

**Janak's theorem** states that $\varepsilon_i = \partial E_\mathrm{tot} / \partial f_i$, so KS eigenvalues are approximate ionization energies under the assumption that the density responds rigidly to a change in occupation. The highest occupied eigenvalue in finite systems equals the exact ionization potential within exact DFT (by Koopmans' theorem for DFT). For extended systems, however, KS eigenvalues are not rigorously quasiparticle energies.

The most practical consequence is the **bandgap problem**: LDA and GGA systematically underestimate semiconductor and insulator bandgaps, typically by 30–50%. The KS gap $\varepsilon_\mathrm{CBM} - \varepsilon_\mathrm{VBM}$ misses the derivative discontinuity of $E_\mathrm{xc}$ at integer electron number. Hybrid functionals (HSE06, PBE0 — already implemented in KRONOS as of v2.0) partially correct this by mixing in exact Fock exchange, reducing the gap error to ~10–15%.

Despite this limitation, KS band structures are qualitatively reliable: band ordering, dispersion, Fermi surface topology, and group velocities are well reproduced by LDA/GGA, making them indispensable for materials screening and property prediction.

## How KRONOS implements this

KRONOS's implementation maps directly onto the KS formalism:

- **`SCFSolver`** (`src/solver/scf.cpp`) drives the SCF loop. It maintains the current density, calls the potential builders, invokes the eigensolver per k-point, applies Fermi-level bisection, and runs Pulay/DIIS mixing.

- **Hamiltonian application** (`src/hamiltonian/`) computes $H|\psi\rangle$ as: (1) kinetic term $|\mathbf{k}+\mathbf{G}|^2 c(\mathbf{G})$ in G-space; (2) local potential $v_\mathrm{eff}(\mathbf{r}) \psi(\mathbf{r})$ via inverse FFT → multiply → FFT; (3) nonlocal pseudopotential via GEMM projections. See [Algorithms](/docs/architecture/algorithms) for the GPU hot-path breakdown.

- **Eigensolver**: Davidson diagonalization at each k-point (subspace dimension $3N_\mathrm{bands}$), with automatic fallback to LOBPCG if the Davidson residual exceeds $10^3$.

- **Potential** (`src/potential/`): Hartree via G-space Poisson ($v_\mathrm{H}(\mathbf{G}) = 8\pi n(\mathbf{G})/|\mathbf{G}|^2$, $G=0$ set to zero); XC via libxc or built-in LDA.

- **k-point parallelism** (`src/utils/mpi_wrapper.cpp`): k-points are distributed round-robin across MPI ranks; densities are reduced via `MPI_Allreduce` before mixing.

The SCF convergence criteria are $|dE| < 10^{-8}$ Ry and $|dn|_\mathrm{max} < 10^{-7}$ e/bohr³, with a hard abort at 200 iterations if neither is met. See [SCF Flowchart](/docs/architecture/scf-flowchart) for the complete decision tree.

## References

- Hohenberg, P. & Kohn, W. "Inhomogeneous electron gas", *Phys. Rev.* **136**, B864 (1964)
- Kohn, W. & Sham, L. J. "Self-consistent equations including exchange and correlation effects", *Phys. Rev.* **140**, A1133 (1965)
- Parr, R. G. & Yang, W. *Density-Functional Theory of Atoms and Molecules*, Oxford University Press, 1989
- Martin, R. M. *Electronic Structure: Basic Theory and Practical Methods*, Cambridge University Press, Ch. 6–7
- Janak, J. F. "Proof that $\partial E / \partial n_i = \varepsilon_i$ in density-functional theory", *Phys. Rev. B* **18**, 7165 (1978)
