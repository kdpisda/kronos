---
title: Bloch's Theorem and Crystal Wavefunctions
description: Bloch's theorem in plane-wave DFT — periodic crystals, Brillouin zone, Monkhorst-Pack k-points, plane-wave expansion of crystal wavefunctions.
keywords:
  - Bloch theorem
  - Bloch states
  - crystal wavefunctions
  - Brillouin zone
  - Monkhorst-Pack k-points
  - plane-wave expansion
  - reciprocal lattice
  - DFT
  - plane-wave DFT
  - k-point sampling
  - crystal momentum
  - irreducible Brillouin zone
slug: /physics/bloch-theorem
sidebar_position: 2
---

# Bloch's Theorem and Crystal Wavefunctions

Bloch's theorem is the cornerstone of electronic structure theory in periodic solids. It converts an apparently intractable problem — finding quantum states in a potential with infinitely many ion cores — into a tractable eigenvalue problem on a single unit cell. Every piece of the KRONOS plane-wave formalism, from G-vector enumeration to k-point grids, is a direct consequence of this theorem.

## Periodic potentials and translation symmetry

A crystal is defined by a Bravais lattice: a set of translation vectors $\mathbf{R} = n_1 \mathbf{a}_1 + n_2 \mathbf{a}_2 + n_3 \mathbf{a}_3$ with $n_i \in \mathbb{Z}$ and primitive lattice vectors $\{\mathbf{a}_i\}$. The ionic potential seen by an electron satisfies

$$V(\mathbf{r} + \mathbf{R}) = V(\mathbf{r}) \quad \text{for all lattice vectors } \mathbf{R}.$$

Because the Hamiltonian $H = -\nabla^2 + V(\mathbf{r})$ inherits this symmetry (in Rydberg units, where the kinetic prefactor is 1), the physics cannot depend on which unit cell we are in. This simple observation has profound consequences.

Define the lattice translation operator $\hat{T}_{\mathbf{R}}$ by its action on a wavefunction:

$$\hat{T}_{\mathbf{R}} \, \psi(\mathbf{r}) = \psi(\mathbf{r} + \mathbf{R}).$$

Periodicity of $V$ means $[H, \hat{T}_{\mathbf{R}}] = 0$: the Hamiltonian commutes with every lattice translation. Moreover, translations compose simply:

$$\hat{T}_{\mathbf{R}} \hat{T}_{\mathbf{R}'} = \hat{T}_{\mathbf{R}+\mathbf{R}'} = \hat{T}_{\mathbf{R}'} \hat{T}_{\mathbf{R}},$$

so the translation operators also commute with each other. We therefore have a complete set of simultaneous eigenstates of $H$ and all $\hat{T}_{\mathbf{R}}$.

The eigenvalue of $\hat{T}_{\mathbf{R}}$ must be a pure phase: since applying $\hat{T}_{\mathbf{R}}$ $N$ times on a finite crystal with periodic boundary conditions returns to the original state, the eigenvalue $\lambda(\mathbf{R})$ satisfies $|\lambda(\mathbf{R})|^N = 1$, so $|\lambda| = 1$. Writing $\lambda(\mathbf{R}) = e^{i\mathbf{k}\cdot\mathbf{R}}$ for some vector $\mathbf{k}$, we arrive at the defining property of a Bloch state:

$$\hat{T}_{\mathbf{R}} \, \psi_{n\mathbf{k}}(\mathbf{r}) = e^{i\mathbf{k}\cdot\mathbf{R}} \, \psi_{n\mathbf{k}}(\mathbf{r}).$$

The vector $\mathbf{k}$ is the **crystal momentum**, and $n$ labels distinct eigenstates (bands) at the same $\mathbf{k}$.

## Statement of Bloch's theorem

**Bloch's theorem:** Every eigenstate of a Hamiltonian with lattice-periodic potential can be written in the form

$$\boxed{\psi_{n\mathbf{k}}(\mathbf{r}) = e^{i\mathbf{k}\cdot\mathbf{r}} \, u_{n\mathbf{k}}(\mathbf{r})}$$

where the **cell-periodic part** $u_{n\mathbf{k}}(\mathbf{r})$ satisfies

$$u_{n\mathbf{k}}(\mathbf{r} + \mathbf{R}) = u_{n\mathbf{k}}(\mathbf{r}) \quad \text{for all } \mathbf{R}.$$

The full wavefunction $\psi_{n\mathbf{k}}$ is a plane wave $e^{i\mathbf{k}\cdot\mathbf{r}}$ modulated by a lattice-periodic envelope $u_{n\mathbf{k}}$. The band index $n$ distinguishes the countably infinite set of solutions at each crystal momentum $\mathbf{k}$.

## Derivation from commuting translations

The derivation is a two-step application of standard linear algebra for commuting operators.

**Step 1.** Since $[H, \hat{T}_{\mathbf{R}}] = 0$, eigenstates of $H$ can be chosen to be simultaneous eigenstates of all $\hat{T}_{\mathbf{R}}$. Call such a state $|\psi_{n\mathbf{k}}\rangle$, with $\hat{T}_{\mathbf{R}}|\psi_{n\mathbf{k}}\rangle = \lambda_{\mathbf{R}} |\psi_{n\mathbf{k}}\rangle$.

**Step 2.** The eigenvalue must be consistent with the group structure of translations. Since $\hat{T}_{\mathbf{R}+\mathbf{R}'} = \hat{T}_{\mathbf{R}}\hat{T}_{\mathbf{R}'}$, the eigenvalues must satisfy $\lambda_{\mathbf{R}+\mathbf{R}'} = \lambda_{\mathbf{R}} \lambda_{\mathbf{R}'}$. The only continuous solutions to this functional equation that satisfy the periodic boundary condition $\lambda_{N_i \mathbf{a}_i} = 1$ are of the form $\lambda_{\mathbf{R}} = e^{i\mathbf{k}\cdot\mathbf{R}}$.

**Step 3.** Define $u_{n\mathbf{k}}(\mathbf{r}) \equiv e^{-i\mathbf{k}\cdot\mathbf{r}} \psi_{n\mathbf{k}}(\mathbf{r})$. Then:

$$u_{n\mathbf{k}}(\mathbf{r}+\mathbf{R}) = e^{-i\mathbf{k}\cdot(\mathbf{r}+\mathbf{R})} \psi_{n\mathbf{k}}(\mathbf{r}+\mathbf{R}) = e^{-i\mathbf{k}\cdot\mathbf{r}} e^{-i\mathbf{k}\cdot\mathbf{R}} \cdot e^{i\mathbf{k}\cdot\mathbf{R}} \psi_{n\mathbf{k}}(\mathbf{r}) = u_{n\mathbf{k}}(\mathbf{r}).$$

So $u_{n\mathbf{k}}$ is lattice-periodic by construction, which proves the Bloch form. The substitution $\psi = e^{i\mathbf{k}\cdot\mathbf{r}} u$ transforms the Kohn-Sham equation $H\psi = \varepsilon\psi$ into a Hermitian eigenvalue problem for $u$ on the unit cell alone, making the problem finite.

## The Brillouin zone

The crystal momentum $\mathbf{k}$ is defined in reciprocal space. The **reciprocal lattice** is spanned by vectors $\mathbf{b}_1, \mathbf{b}_2, \mathbf{b}_3$ satisfying $\mathbf{a}_i \cdot \mathbf{b}_j = 2\pi \delta_{ij}$:

$$\mathbf{G} = m_1 \mathbf{b}_1 + m_2 \mathbf{b}_2 + m_3 \mathbf{b}_3, \quad m_i \in \mathbb{Z}.$$

A crucial observation: $\mathbf{k}$ and $\mathbf{k} + \mathbf{G}$ label physically equivalent states. To see this, note that $e^{i\mathbf{G}\cdot\mathbf{R}} = 1$ for any reciprocal lattice vector $\mathbf{G}$ and any lattice vector $\mathbf{R}$, so both $\mathbf{k}$ and $\mathbf{k}+\mathbf{G}$ produce the same translation eigenvalue $e^{i\mathbf{k}\cdot\mathbf{R}}$. Consequently:

$$\psi_{n,\mathbf{k}+\mathbf{G}}(\mathbf{r}) = \psi_{n'\mathbf{k}}(\mathbf{r}) \quad \text{(same physical state, different band label } n').$$

This redundancy means $\mathbf{k}$ need only be sampled within one primitive cell of the reciprocal lattice. The standard choice is the **first Brillouin zone (BZ)**: the Wigner-Seitz cell of the reciprocal lattice, i.e., the set of all points in reciprocal space closer to $\mathbf{G} = 0$ than to any other reciprocal lattice vector. Its volume is

$$\Omega_\mathrm{BZ} = \frac{(2\pi)^3}{\Omega_\mathrm{cell}},$$

where $\Omega_\mathrm{cell} = |\mathbf{a}_1 \cdot (\mathbf{a}_2 \times \mathbf{a}_3)|$ is the real-space unit cell volume. Every distinct crystal momentum $\mathbf{k}$ is represented exactly once in the first BZ.

## Plane-wave expansion

Since $u_{n\mathbf{k}}(\mathbf{r})$ is lattice-periodic, it can be expanded exactly in a Fourier series over reciprocal lattice vectors:

$$u_{n\mathbf{k}}(\mathbf{r}) = \sum_{\mathbf{G}} c_{n\mathbf{k},\mathbf{G}} \, e^{i\mathbf{G}\cdot\mathbf{r}}.$$

Substituting back into the Bloch form gives the **plane-wave expansion of the Bloch wavefunction**:

$$\psi_{n\mathbf{k}}(\mathbf{r}) = \sum_{\mathbf{G}} c_{n\mathbf{k},\mathbf{G}} \, e^{i(\mathbf{k}+\mathbf{G})\cdot\mathbf{r}}.$$

The plane waves $\{e^{i(\mathbf{k}+\mathbf{G})\cdot\mathbf{r}}\}$ form a complete, orthonormal basis on the unit cell:

$$\frac{1}{\Omega} \int_\mathrm{cell} e^{-i(\mathbf{k}+\mathbf{G})\cdot\mathbf{r}} e^{i(\mathbf{k}+\mathbf{G}')\cdot\mathbf{r}} d\mathbf{r} = \delta_{\mathbf{G},\mathbf{G}'}.$$

In this basis, the Kohn-Sham equation becomes a matrix eigenvalue problem. The kinetic energy is diagonal:

$$\langle \mathbf{k}+\mathbf{G} | T | \mathbf{k}+\mathbf{G}' \rangle = |\mathbf{k}+\mathbf{G}|^2 \, \delta_{\mathbf{G},\mathbf{G}'}$$

(in Rydberg units, where $T = -\nabla^2$). The local potential couples G-vectors through its Fourier components: $\langle \mathbf{k}+\mathbf{G} | V | \mathbf{k}+\mathbf{G}' \rangle = \tilde{V}(\mathbf{G}-\mathbf{G}')$.

**Basis truncation.** In practice, the sum over $\mathbf{G}$ is truncated by an energy cutoff `ecutwfc`:

$$|\mathbf{k}+\mathbf{G}|^2 \leq E_\mathrm{cut}.$$

This retains the kinetically most important plane waves and systematically converges to the exact result as $E_\mathrm{cut} \to \infty$. A typical converged cutoff is 30–100 Ry depending on the pseudopotential. KRONOS stores the coefficients $c_{n\mathbf{k},\mathbf{G}}$ as `complex128` (double-precision complex) vectors — no float32 short-cuts.

The density $n(\mathbf{r}) = \sum_{n\mathbf{k}} f_{n\mathbf{k}} |\psi_{n\mathbf{k}}(\mathbf{r})|^2$ involves products of two wavefunctions, so its Fourier components extend to $|2\mathbf{G}|^2 \leq 4E_\mathrm{cut}$. KRONOS therefore requires `ecutrho` $\geq 4 \times$ `ecutwfc` for norm-conserving pseudopotentials to represent the density without aliasing.

## K-point sampling: Monkhorst-Pack and the IBZ

Physical observables involve integrals over the Brillouin zone. For example, the electron density is

$$n(\mathbf{r}) = \frac{2}{\Omega_\mathrm{BZ}} \int_\mathrm{BZ} \sum_n f_{n\mathbf{k}} \, |\psi_{n\mathbf{k}}(\mathbf{r})|^2 \, d\mathbf{k},$$

where the factor of 2 accounts for spin degeneracy (spinless case). In practice this integral is replaced by a discrete sum:

$$\frac{1}{\Omega_\mathrm{BZ}} \int_\mathrm{BZ} F(\mathbf{k}) \, d\mathbf{k} \;\longrightarrow\; \sum_{\mathbf{k} \in \mathrm{grid}} w_{\mathbf{k}} \, F(\mathbf{k})$$

with weights $w_{\mathbf{k}}$ summing to 1. The **Monkhorst-Pack** scheme generates a uniform grid by choosing

$$\mathbf{k}_{n_1 n_2 n_3} = \sum_{i=1}^{3} \frac{2n_i - N_i - 1 + s_i}{2N_i} \, \mathbf{b}_i, \quad n_i = 1, \ldots, N_i,$$

where $N_i$ is the number of grid points along reciprocal axis $i$ and $s_i \in \{0, 1\}$ is an optional shift. An unshifted grid ($s_i = 0$) always includes the $\Gamma$ point; a shifted grid ($s_i = 1$) avoids it and can converge faster for metals.

**Time-reversal symmetry.** In a non-magnetic crystal with time-reversal symmetry, $\varepsilon_{n,-\mathbf{k}} = \varepsilon_{n,\mathbf{k}}$. This allows $\mathbf{k}$ and $-\mathbf{k}$ to be folded together, roughly halving the number of k-points that must be computed explicitly.

**Irreducible Brillouin zone (IBZ).** The crystal's full space-group symmetry (rotations and screw/glide operations) further reduces the grid. Two k-points related by a symmetry operation $S$ of the point group satisfy $\varepsilon_{n,S\mathbf{k}} = \varepsilon_{n,\mathbf{k}}$, so only one representative from each symmetry-equivalent set — the **irreducible Brillouin zone** — needs to be computed. Each IBZ k-point $\mathbf{k}_i$ carries a weight $w_i = N_\mathrm{equiv}(\mathbf{k}_i) / N_\mathrm{total}$, where $N_\mathrm{equiv}$ is the size of its star. For example, face-centered cubic Si with a $4\times4\times4$ Monkhorst-Pack grid has 64 k-points in the full BZ but only 10 in the IBZ — a 6$\times$ reduction.

## How KRONOS implements this

**K-point generation.** KRONOS generates Monkhorst-Pack grids from the `kpoints` block in the YAML input:

```yaml
kpoints:
  grid: [4, 4, 4]
  shift: [1, 1, 1]   # 0 or 1 per axis
```

When `spglib` is linked, KRONOS calls it to detect the full space group of the crystal, obtain all symmetry operations, and fold the grid down to the IBZ. Without spglib, only time-reversal folding ($\mathbf{k} \leftrightarrow -\mathbf{k}$) is applied.

**`PlaneWaveBasis`.** Once the IBZ k-points are known, KRONOS builds a single shared G-vector basis. For each k-point $\mathbf{k}_i$, it finds all $\mathbf{G}$ such that $|\mathbf{k}_i + \mathbf{G}|^2 \leq E_\mathrm{cut}$. The union of all per-k sets forms the shared basis stored in `PlaneWaveBasis::gvectors()`. This avoids allocating a separate basis per k-point and allows the Davidson solver to operate on a single block of memory.

When the Hamiltonian is applied at a specific k-point, G-vectors outside that k-point's active set are masked: their kinetic energy is set to a hard wall of $10^4$ Ry, driving the Davidson solver to assign them zero amplitude without requiring a dynamically sized array. This is a deliberate performance/correctness trade-off — a fixed-size layout with an energy mask is simpler and cache-friendlier than per-k variable-length slices.

**`KPoints` class.** The k-point list, IBZ weights, and per-k active G-vector masks are stored in the `KPoints` object (`src/basis/kpoints.hpp`). The SCF loop in `src/solver/scf.cpp` iterates over `kpoints.ibz()` — the list of irreducible k-points — applying $H|\psi\rangle$ and accumulating the density with the correct $w_\mathbf{k}$ weights. After eigenvalues are collected across all k-points, the Fermi level is determined by bisection and occupations $f_{n\mathbf{k}}$ are assigned, closing the k-point loop for that SCF step.

**Wavefunction storage.** For each IBZ k-point, KRONOS stores the $N_\mathrm{bands} \times N_\mathrm{pw}$ coefficient matrix $c_{n\mathbf{k}}(\mathbf{G})$ as a contiguous block of `complex_t` (which aliases `std::complex<double>`). This layout is chosen for BLAS compatibility: the nonlocal pseudopotential application uses cuBLAS/rocBLAS GEMM on this matrix, with the beta projectors $\beta_i(\mathbf{k}+\mathbf{G})$ as the other factor.

## References

- Martin, R. M. *Electronic Structure: Basic Theory and Practical Methods*, Cambridge University Press, Ch. 4 (2004)
- Ashcroft, N. W. & Mermin, N. D. *Solid State Physics*, Holt, Rinehart and Winston, Ch. 8 (1976)
- Monkhorst, H. J. & Pack, J. D. "Special points for Brillouin-zone integrations", *Phys. Rev. B* **13**, 5188 (1976). [DOI:10.1103/PhysRevB.13.5188](https://doi.org/10.1103/PhysRevB.13.5188)
- Bloch, F. "Über die Quantenmechanik der Elektronen in Kristallgittern", *Z. Phys.* **52**, 555 (1929)
- Payne, M. C. et al. "Iterative minimization techniques for ab initio total-energy calculations: molecular dynamics and conjugate gradients", *Rev. Mod. Phys.* **64**, 1045 (1992). [DOI:10.1103/RevModPhys.64.1045](https://doi.org/10.1103/RevModPhys.64.1045)
