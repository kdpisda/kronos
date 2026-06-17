---
title: Plane-Wave Basis and Energy Cutoffs
description: How KRONOS represents wavefunctions as plane waves, kinetic-energy cutoff conventions (ecutwfc, ecutrho), FFT grid sizing, and DFT convergence behavior.
keywords:
  - plane-wave basis
  - kinetic energy cutoff
  - ecutwfc
  - ecutrho
  - FFT grid
  - plane-wave DFT
  - DFT convergence
  - reciprocal lattice
  - dual representation
  - G-vector enumeration
  - Bloch theorem
  - wavefunction expansion
slug: /physics/plane-wave-basis
sidebar_position: 3
---

# Plane-Wave Basis and Energy Cutoffs

Plane waves are the natural basis for periodic systems: they are orthonormal, unbiased across the unit cell, and make both the kinetic energy and the Hartree potential diagonal in reciprocal space. KRONOS expands all Kohn-Sham wavefunctions in a plane-wave basis truncated by a kinetic-energy cutoff `ecutwfc`, stores the coefficients as `complex128` vectors, and uses a dual real/reciprocal representation — connected by FFT — to apply each part of the Hamiltonian where it is cheapest.

## Why plane waves

A crystalline solid has discrete translational symmetry: the electron density and the effective potential repeat with the Bravais lattice vectors $\mathbf{T}$. By Bloch's theorem, the single-particle orbitals take the form

$$\psi_{n\mathbf{k}}(\mathbf{r}) = e^{i\mathbf{k}\cdot\mathbf{r}} \, u_{n\mathbf{k}}(\mathbf{r})$$

where $u_{n\mathbf{k}}$ has the full lattice periodicity. Because $u_{n\mathbf{k}}$ is periodic, it can be expanded exactly in reciprocal lattice vectors $\mathbf{G}$, which are the Fourier modes of the lattice. The natural basis is therefore the set of plane waves $e^{i(\mathbf{k}+\mathbf{G})\cdot\mathbf{r}}$, and this is not an arbitrary choice — it is mandated by the symmetry.

Three properties make plane waves uniquely attractive for DFT:

1. **Orthonormality.** $\langle \mathbf{G} | \mathbf{G}' \rangle = \delta_{\mathbf{G},\mathbf{G}'}$ exactly, with no overlap matrix to invert. Gram-Schmidt orthogonalization is trivial.

2. **Uniformity.** Every plane wave samples the entire unit cell equally. There is no basis-set superposition error and no center-bias around atomic sites. Adding or removing atoms does not change the basis — only the cutoff controls completeness.

3. **Diagonal operators.** The kinetic energy $T = -\nabla^2$ is diagonal in G-space: $T_\mathbf{G} = |\mathbf{k}+\mathbf{G}|^2$ (Rydberg units). The Hartree potential is diagonal in G-space too: $V_H(\mathbf{G}) = 8\pi n(\mathbf{G})/|\mathbf{G}|^2$. Local potentials become pointwise multiplications in real space. No integrals need to be evaluated explicitly — everything is either a diagonal multiply or an FFT.

## Defining the basis

The complete plane-wave basis for Bloch vector $\mathbf{k}$ is

$$\phi_{\mathbf{k}+\mathbf{G}}(\mathbf{r}) = \frac{1}{\sqrt{\Omega}} \, e^{i(\mathbf{k}+\mathbf{G})\cdot\mathbf{r}}$$

where $\Omega$ is the unit cell volume and $\mathbf{G}$ runs over all reciprocal lattice vectors $\mathbf{G} = h\mathbf{b}_1 + k\mathbf{b}_2 + l\mathbf{b}_3$ with integer Miller indices $(h, k, l)$. The normalization $1/\sqrt{\Omega}$ ensures $\langle \phi_{\mathbf{k}+\mathbf{G}} | \phi_{\mathbf{k}+\mathbf{G}'} \rangle = \delta_{\mathbf{G},\mathbf{G}'}$.

The Kohn-Sham wavefunction for band $n$ at k-point $\mathbf{k}$ is then

$$\psi_{n\mathbf{k}}(\mathbf{r}) = \frac{1}{\sqrt{\Omega}} \sum_{\mathbf{G}} c_{n\mathbf{k}}(\mathbf{G}) \, e^{i(\mathbf{k}+\mathbf{G})\cdot\mathbf{r}}$$

KRONOS stores and manipulates these coefficients $c_{n\mathbf{k}}(\mathbf{G})$ as `complex128` (double-precision complex) vectors. The basis is complete only in the limit of infinitely many $\mathbf{G}$ vectors — in practice it is truncated by a kinetic-energy cutoff.

## Kinetic energy cutoff

The truncation criterion is the kinetic energy of each plane wave. In **Rydberg units** ($\hbar = 1$, $m_e = 1/2$), the kinetic-energy operator is $T = -\nabla^2$, so the kinetic energy of the plane wave $e^{i(\mathbf{k}+\mathbf{G})\cdot\mathbf{r}}$ is $|\mathbf{k}+\mathbf{G}|^2$ — note the absence of the $1/2$ prefactor that would appear in Hartree units.

The wavefunction basis is then the sphere of G-vectors satisfying

$$|\mathbf{k}+\mathbf{G}|^2 \leq E_\mathrm{cut}^{\mathrm{wfc}}$$

The KRONOS YAML key is `ecutwfc` (in Ry). This defines a sphere of radius $\sqrt{E_\mathrm{cut}^{\mathrm{wfc}}}$ centered at $-\mathbf{k}$ in reciprocal space. G-vectors inside the sphere are included; those outside are discarded.

The number of plane waves $N_\mathrm{pw}$ grows as $\Omega \cdot (E_\mathrm{cut}^{\mathrm{wfc}})^{3/2}$ — doubling the cutoff roughly triples the basis size, and therefore the cost of each matrix-vector product $H|\psi\rangle$ and each FFT.

:::note Rydberg vs Hartree units
In Hartree atomic units the cutoff condition is $\frac{1}{2}|\mathbf{k}+\mathbf{G}|^2 \leq E_\mathrm{cut}^{\mathrm{Ha}}$, so $E_\mathrm{cut}^{\mathrm{Ry}} = 2 \, E_\mathrm{cut}^{\mathrm{Ha}}$. KRONOS uses Rydberg throughout; a cutoff of 40 Ry in KRONOS corresponds to 20 Ha in Quantum ESPRESSO's `ecutwfc`. All physics notes in this documentation use Rydberg units unless stated otherwise.
:::

## Per-k masks and the global G-sphere

Each k-point $\mathbf{k}$ has its own cutoff sphere shifted to $-\mathbf{k}$ in G-space, so the active set of G-vectors differs between k-points. Maintaining a separate G-vector list for each k-point simplifies some operations but complicates MPI data layout and memory management.

KRONOS uses a different strategy: a single **global G-sphere** expanded to cover all k-points simultaneously. Every G-vector satisfying

$$|\mathbf{k}+\mathbf{G}|^2 \leq E_\mathrm{cut}^{\mathrm{wfc}} \quad \text{for any } \mathbf{k} \in \mathrm{IBZ}$$

is included in the global list. In practice this means the sphere is expanded by $k_\mathrm{max} = \max_\mathbf{k} |\mathbf{k}|$, which is a modest overhead for typical Monkhorst-Pack grids.

When applying $H|\psi\rangle$ at a specific k-point, a **per-k mask** zeros out the G-vectors outside that k-point's individual sphere and assigns them a large kinetic energy (the `energy_wall` = $10^4$ Ry). This drives the Davidson solver to collapse those amplitudes to zero without any special-casing in the eigensolver logic.

The data flow is: `PlaneWaveBasis` holds the full G-vector list and `|k+G|^2` values; `Hamiltonian` receives the mask per k-point at the start of each diagonalization; the Davidson solver sees a fixed-size vector and works with the mask in place.

## Density cutoff and the FFT grid

The electron density is constructed from the wavefunctions:

$$n(\mathbf{r}) = \sum_{n\mathbf{k}} f_{n\mathbf{k}} \, |\psi_{n\mathbf{k}}(\mathbf{r})|^2$$

Since $\psi(\mathbf{r})$ contains components up to $|\mathbf{G}|_\mathrm{max}^\mathrm{wfc} = \sqrt{E_\mathrm{cut}^\mathrm{wfc}}$, the product $|\psi|^2$ contains components up to twice that frequency. To represent the density without aliasing, the density grid must accommodate G-vectors up to $2\sqrt{E_\mathrm{cut}^\mathrm{wfc}}$, which corresponds to a cutoff four times larger:

$$E_\mathrm{cut}^\mathrm{rho} \geq 4 \, E_\mathrm{cut}^\mathrm{wfc} \quad \text{(norm-conserving pseudopotentials)}$$

For PAW (projector-augmented wave) pseudopotentials, the augmentation charges are sharper than the pseudo-density, requiring an even larger grid:

$$E_\mathrm{cut}^\mathrm{rho} \geq 12 \, E_\mathrm{cut}^\mathrm{wfc} \quad \text{(PAW)}$$

The KRONOS YAML key is `ecutrho`. If `ecutrho` is not specified, KRONOS defaults to $4 \times$ `ecutwfc` for NCPP and $12 \times$ for PAW. Setting it below these limits is a hard error.

**FFT grid sizing.** The full FFT grid that holds the density and potentials must be large enough to represent all G-vectors up to $\sqrt{E_\mathrm{cut}^\mathrm{rho}}$. For a simulation cell with lattice vector $L_i$ in direction $i$, the minimum grid dimension is

$$N_i \geq \frac{L_i}{\pi} \cdot \sqrt{E_\mathrm{cut}^\mathrm{rho}}$$

This follows from the Nyquist condition: the grid spacing $\Delta x_i = L_i / N_i$ must satisfy $\pi/\Delta x_i \geq G_\mathrm{max}$, so $N_i \geq L_i G_\mathrm{max} / \pi$.

KRONOS then rounds $N_i$ up to the next integer whose prime factors are all in $\{2, 3, 5\}$ — so-called "FFT-friendly" numbers. FFTW3 (and cuFFT) achieve near-optimal performance on these composites; arbitrary large primes in $N_i$ can be orders of magnitude slower. In practice this means $N_i$ is always of the form $2^a \cdot 3^b \cdot 5^c$.

## Real-space and reciprocal-space dual

The power of plane waves lies in being able to work in whichever representation makes each operation cheap. Two operations are diagonal in complementary spaces:

| Operation | Optimal space | Cost |
|-----------|--------------|------|
| Kinetic energy: $T|\psi\rangle = |\mathbf{k}+\mathbf{G}|^2 \, c(\mathbf{G})$ | G-space | $O(N_\mathrm{pw})$ |
| Local potential: $V|\psi\rangle = V_\mathrm{eff}(\mathbf{r}) \cdot \psi(\mathbf{r})$ | Real space | $O(N_\mathrm{grid})$ |

Switching between representations costs $O(N \log N)$ via FFT. The application of the local potential is therefore:

1. IFFT $\psi(\mathbf{G}) \to \psi(\mathbf{r})$ on the FFT grid
2. Pointwise multiply $V_\mathrm{eff}(\mathbf{r}) \cdot \psi(\mathbf{r})$
3. FFT back $(V_\mathrm{eff}\psi)(\mathbf{r}) \to (V_\mathrm{eff}\psi)(\mathbf{G})$

The nonlocal pseudopotential contribution bypasses this entirely and uses BLAS GEMM in G-space. The full Hamiltonian application $H|\psi\rangle$ is thus a composition of three operations:

$$H|\psi\rangle = \underbrace{|\mathbf{k}+\mathbf{G}|^2 \, c(\mathbf{G})}_{\text{kinetic (G-space)}} + \underbrace{\mathrm{FFT}\!\left[V_\mathrm{eff}(\mathbf{r}) \cdot \mathrm{IFFT}[\psi(\mathbf{G})]\right]}_{\text{local (real-space round-trip)}} + \underbrace{\sum_{a,ij} |\beta_i^a\rangle D_{ij}^a \langle\beta_j^a|\psi\rangle}_{\text{nonlocal (G-space GEMM)}}$$

The FFT round-trip is the dominant cost for large cutoffs — it is the hot path on both CPU and GPU. See [Data Flow Through the SCF Loop](../architecture/data-flow.md) for how `FFTGrid` coordinates the `scatter_to_grid` and `gather_from_grid` operations that move between the wavefunction PW basis and the full density grid.

## Convergence in `ecutwfc`

Total energy is a **variational** functional of the wavefunction: adding more plane waves (increasing `ecutwfc`) can only lower the energy. This means energy converges monotonically from above as `ecutwfc` increases — it never overshoots the converged value.

Convergence rate depends strongly on the pseudopotential. Smooth norm-conserving PPs (e.g., ONCV) typically converge by 40–60 Ry. Harder PPs — or all-electron calculations — may require 80–200 Ry. PAW pseudopotentials decouple the wavefunction smoothness from the augmentation density and often converge faster in the wavefunction cutoff.

**Convergence test recipe.** The standard practice is to vary `ecutwfc` over a range (e.g., 20, 30, 40, 50, 60 Ry) and check that the total energy difference between successive values falls below your target precision $\varepsilon$:

$$|E(E_\mathrm{cut}) - E(E_\mathrm{cut} + \Delta)| < \varepsilon$$

A common target for structural properties is $\varepsilon = 1$ mRy/atom (0.5 meV/atom). For forces and stress, the convergence is slower — use at least $2\times$ the cutoff that converges the energy. For the KRONOS Delta test benchmark (< 2 meV/atom), a final cutoff check at both `ecutwfc` and `ecutwfc + 10 Ry` is recommended.

:::tip Typical cutoff values
- Si (ONCV): 40 Ry converged to < 0.1 mRy/atom
- Cu (ONCV, semicore 3d): 60–80 Ry
- Fe (ONCV, semicore 3d): 60–80 Ry
- H₂O (ONCV): 40–50 Ry
- MgO (ONCV): 60 Ry

These are guidelines for ONCV pseudopotentials. Norm-conserving PPs from other libraries may differ substantially.
:::

## How KRONOS implements this

**`PlaneWaveBasis::enumerate()`** (`src/basis/`) iterates over integer triplets $(h, k, l)$ in a box large enough to contain the sphere of radius $\sqrt{E_\mathrm{cut}^\mathrm{wfc}} + k_\mathrm{max}$, computes the Cartesian G-vector for each Miller index via the reciprocal lattice matrix, and keeps those satisfying the expanded cutoff condition. The surviving G-vectors are stored in a sorted flat array together with their `|k+G|^2` values for each k-point.

**Per-k masks** are Boolean vectors (one per k-point) stored alongside the global G-list. `Hamiltonian::apply()` selects the appropriate mask before each diagonalization step; the Davidson solver operates on full-length vectors throughout.

**`FFTGrid`** (`src/basis/`) computes the density grid dimensions by applying the Nyquist formula for `ecutrho` and rounding up to the nearest FFT-friendly composite. It then creates FFTW3 plans (or cuFFT plans in GPU builds) in the constructor so that subsequent transforms pay no planning cost. The `scatter_to_grid` method fills the 3D FFT array from the 1D G-vector coefficient array using stored Miller-index-to-grid-index maps; `gather_from_grid` does the reverse.

**GPU note.** On CUDA/HIP builds, FFT operations are performed by cuFFT/rocFFT and BLAS GEMMs by cuBLAS/rocBLAS, all within the `gpu::` abstraction layer (`src/gpu/`). The `PlaneWaveBasis` and `FFTGrid` objects themselves are CPU-resident; only the coefficient arrays and grid buffers are pinned or device-allocated as needed.

## References

- Martin, R. M. *Electronic Structure: Basic Theory and Practical Methods*, Cambridge University Press, Ch. 12 (2004)
- Payne, M. C., Teter, M. P., Allan, D. C., Arias, T. A., Joannopoulos, J. D. "Iterative minimization techniques for ab initio total-energy calculations: molecular dynamics and conjugate gradients", *Rev. Mod. Phys.* **64**, 1045 (1992)
- Blöchl, P. E. "Generalized separable potentials for electronic-structure calculations", *Phys. Rev. B* **41**, 5414 (1990)
- Frigo, M. and Johnson, S. G. "The design and implementation of FFTW3", *Proc. IEEE* **93**, 216 (2005)
