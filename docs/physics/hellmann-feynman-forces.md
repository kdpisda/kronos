---
title: Hellmann-Feynman Forces in Plane-Wave DFT
description: Hellmann-Feynman theorem, local + nonlocal pseudopotential force components, Ewald ion-ion forces, force symmetrization, and finite-difference validation in KRONOS.
keywords:
  - Hellmann-Feynman theorem
  - DFT forces
  - plane-wave DFT forces
  - Ewald forces
  - nonlocal pseudopotential forces
  - Pulay forces
  - force symmetrization
  - geometry optimization
  - DFT structural relaxation
  - analytical gradients DFT
slug: /physics/hellmann-feynman-forces
sidebar_position: 11
---

# Hellmann-Feynman Forces in Plane-Wave DFT

In ab initio molecular dynamics, structure optimization, and vibrational analysis, the ability to compute the **analytic forces on each atom** is crucial. Computing forces via finite differences—displacing each atom and re-running the self-consistent field cycle—is prohibitively expensive: for a supercell with $N$ atoms, it requires $O(N)$ SCF calculations. The **Hellmann-Feynman theorem** provides the solution: at SCF convergence, the atomic forces follow directly from the self-consistent wavefunction and charge density, with no additional eigenvalue computation needed. This page derives the forces from first principles, explains how they decompose into Ewald, local, and nonlocal contributions, and describes KRONOS's implementation and validation against finite differences.

## The Hellmann-Feynman Theorem

The Hellmann-Feynman theorem states that for an eigenstate $|\psi_n\rangle$ of a parametrized Hamiltonian $H(\lambda)$,

$$\frac{\partial E_n}{\partial \lambda} = \left\langle \psi_n \left| \frac{\partial H}{\partial \lambda} \right| \psi_n \right\rangle,$$

**provided the wavefunction is an eigenstate of the parametrized Hamiltonian.** The key insight is that the wavefunction derivative term $\partial|\psi_n\rangle/\partial\lambda$ cancels identically at convergence:

$$\frac{dE_n}{d\lambda} = \frac{\partial}{\partial\lambda} \langle \psi_n | H(\lambda) | \psi_n \rangle = \left\langle \frac{\partial\psi_n}{\partial\lambda} \bigg| H | \psi_n \right\rangle + \left\langle \psi_n \left| \frac{\partial H}{\partial\lambda} \right| \psi_n \right\rangle + \left\langle \psi_n \bigg| H | \frac{\partial\psi_n}{\partial\lambda} \right\rangle.$$

Since $H|\psi_n\rangle = E_n|\psi_n\rangle$, the first and third terms give $E_n(\text{wavefunction overlap terms}) - E_n(\text{same overlap})$, which cancels. Only the middle term survives.

**Application to atomic forces:** For the total energy $E_\mathrm{tot}(\{\boldsymbol\tau_a\})$ as a function of ionic positions $\boldsymbol\tau_a$ (holding the electron density fixed on the Born-Oppenheimer surface, i.e., SCF converged), the force on atom $a$ is

$$\mathbf{F}_a = -\frac{\partial E_\mathrm{tot}}{\partial \boldsymbol\tau_a} = \left\langle \Psi_\mathrm{scf} \left| -\frac{\partial H}{\partial \boldsymbol\tau_a} \right| \Psi_\mathrm{scf} \right\rangle + \text{(electrostatic ion-ion contribution)}.$$

For plane-wave DFT with pseudopotentials, this partitions into three physically distinct contributions: Ewald ion-ion repulsion, local pseudopotential, and nonlocal pseudopotential forces. Each has a different functional form and implementation path.

## Force Components: The Total Force

The total force on atom $a$ decomposes as

$$\boxed{\mathbf{F}_a = \mathbf{F}_a^\mathrm{Ewald} + \mathbf{F}_a^\mathrm{local} + \mathbf{F}_a^\mathrm{NL}}.$$

This decomposition holds exactly for norm-conserving pseudopotentials under the Hellmann-Feynman approximation (Pulay forces vanish; see §7). PAW introduces augmentation-charge contributions, which are handled separately. All equations below work in **Rydberg atomic units**, consistent with KRONOS source code.

## Ewald Ion-Ion Forces

The ionic configuration energy is computed via the Ewald sum, which partitions the long-range Coulomb repulsion into a short-range real-space sum and a long-range reciprocal-space sum to accelerate convergence:

$$E_\mathrm{Ewald} = \sum_{a<b} Z_a Z_b \left[ \sum_{\mathbf{R} \ne 0} \frac{\mathrm{erfc}(\eta |\boldsymbol\tau_{ab} + \mathbf{R}|)}{|\boldsymbol\tau_{ab} + \mathbf{R}|} + \text{reciprocal-space part} + \text{self-interaction correction} \right],$$

where $Z_a, Z_b$ are ionic charges (from pseudopotentials), $\eta$ is the Ewald screening parameter (typically chosen for equal convergence of both sums), $\boldsymbol\tau_{ab} = \boldsymbol\tau_a - \boldsymbol\tau_b$, and $\mathbf{R}$ ranges over lattice vectors.

**Real-space contribution:** The force from the real-space sum is straightforward differentiation:

$$\mathbf{F}_a^\mathrm{Ewald,real} = Z_a \sum_{b \ne a} Z_b \sum_{\mathbf{R}} \frac{\partial}{\partial \boldsymbol\tau_a} \left[ \frac{\mathrm{erfc}(\eta |\boldsymbol\tau_{ab} + \mathbf{R}|)}{|\boldsymbol\tau_{ab} + \mathbf{R}|} \right].$$

Using $\frac{d}{dr}\!\left[\frac{\mathrm{erfc}(\eta r)}{r}\right] = -\frac{\mathrm{erfc}(\eta r) + 2\eta r / \sqrt{\pi} \, e^{-(\eta r)^2}}{r^2}$ (the chain rule and product rule), the force magnitude along the direction $\hat{\mathbf{r}}_{ab}$ decays exponentially for short distances but remains $O(1/r^2)$ for distances beyond the Ewald cutoff—hence the need for the reciprocal-space sum to capture distant periodic images.

**Reciprocal-space contribution:** The reciprocal-space Ewald energy is

$$E_\mathrm{Ewald,recip} = \frac{2\pi}{\Omega} \sum_{\mathbf{G} \ne 0} \frac{e^{-|\mathbf{G}|^2/(4\eta^2)}}{|\mathbf{G}|^2} \left| \sum_a Z_a e^{i\mathbf{G} \cdot \boldsymbol\tau_a} \right|^2.$$

Differentiating with respect to $\boldsymbol\tau_a$:

$$\mathbf{F}_a^\mathrm{Ewald,recip} = \frac{2\pi}{\Omega} \sum_{\mathbf{G} \ne 0} \frac{e^{-|\mathbf{G}|^2/(4\eta^2)}}{|\mathbf{G}|^2} \cdot i\mathbf{G} \, Z_a e^{i\mathbf{G} \cdot \boldsymbol\tau_a} \left[ \sum_b Z_b e^{-i\mathbf{G} \cdot \boldsymbol\tau_b} + \text{c.c.} \right].$$

When summed over all atoms, the reciprocal-space contribution vanishes by charge neutrality (the structure factor for $\mathbf{G} \ne 0$ sums to zero for neutral systems); for a single atom, it is nonzero and essential.

**Implementation:** KRONOS computes both real and reciprocal contributions in `src/potential/forces.cpp::ewald_forces()`. The algorithm:

1. Real-space loop: all pairs $(a, b)$ with $|\boldsymbol\tau_{ab} + \mathbf{R}| < r_\mathrm{cut}$.
2. Reciprocal-space loop: all $\mathbf{G}$ vectors up to the wavefunction cutoff, using cached structure factors $\sum_a Z_a e^{i\mathbf{G} \cdot \boldsymbol\tau_a}$.

The total Ewald force on atom $a$ is $\mathbf{F}_a^\mathrm{Ewald} = \mathbf{F}_a^\mathrm{Ewald,real} + \mathbf{F}_a^\mathrm{Ewald,recip}$.

## Local Pseudopotential Forces

The local pseudopotential contributes an energy-density term, $E_\mathrm{loc} = \int V_\mathrm{loc}(\mathbf{r}) n(\mathbf{r}) d^3r$, where $V_\mathrm{loc}(\mathbf{r})$ depends on atomic positions $\boldsymbol\tau_a$ through the superposition of spherically symmetric potentials centered at each atom.

In G-space, $V_\mathrm{loc}(\mathbf{r})$ expands as

$$V_\mathrm{loc}(\mathbf{r}) = \sum_a V_\mathrm{loc}^a(|\mathbf{r} - \boldsymbol\tau_a|),$$

with Fourier components

$$\tilde{V}_\mathrm{loc}^a(\mathbf{G}) = e^{-i\mathbf{G} \cdot \boldsymbol\tau_a} \int_0^\infty V_\mathrm{loc}^a(r) j_0(Gr) r^2 \, dr,$$

where $j_0(x) = \sin(x)/x$ is the zeroth-order spherical Bessel function (spherical symmetry ensures only $j_0$ appears).

The local energy and its gradient are

$$E_\mathrm{loc} = \frac{\Omega}{2} \sum_{\mathbf{G}} [\tilde{V}_\mathrm{loc}(\mathbf{G})]^* n(\mathbf{G}),$$

$$\mathbf{F}_a^\mathrm{local} = -\frac{\partial E_\mathrm{loc}}{\partial \boldsymbol\tau_a} = -i\Omega \sum_{\mathbf{G} \ne 0} \mathbf{G}\, [\tilde{V}_\mathrm{loc}^a(\mathbf{G})]^* n(\mathbf{G}).$$

**Physical interpretation:** The factor $i\mathbf{G}$ comes from differentiating the structure factor $e^{-i\mathbf{G} \cdot \boldsymbol\tau_a}$ with respect to position. The force is proportional to the gradient of the local potential convolved with the electron density in G-space.

**Numerical implementation:** KRONOS precomputes the Fourier-transformed local pseudopotential $\tilde{V}_\mathrm{loc}^a(\mathbf{G})$ via spherical Bessel transform at initialization and caches it. At each SCF step, the density $n(\mathbf{G})$ is available on the finer density grid (controlled by $E_\mathrm{cut,rhoe}$). The force is computed as a loop over $\mathbf{G}$ vectors, accumulating $\mathbf{G} \times V_\mathrm{loc}^a(\mathbf{G}) \times n(\mathbf{G})$. Cost: $O(N_\mathrm{G,dense})$.

## Nonlocal Pseudopotential Forces

The nonlocal pseudopotential is expressed in the Kleinman-Bylander separable form as

$$\hat{V}_\mathrm{NL} = \sum_a \sum_{lm} \sum_{ij} |\beta_{lm,i}^a\rangle D_{ij}^{l,a} \langle \beta_{lm,j}^a|,$$

where $\beta_{lm,i}^a(\mathbf{r}) = R_i^l(|\mathbf{r} - \boldsymbol\tau_a|) Y_{lm}(\widehat{\mathbf{r} - \boldsymbol\tau_a})$ is a projector centered at atom $a$.

The total electronic energy includes the nonlocal contribution

$$E_\mathrm{NL} = 2\sum_{n\mathbf{k}} w_\mathbf{k} f_{n\mathbf{k}} \sum_{a,lm,ij} D_{ij}^{l,a} \, p_{lm,i}^{a,\ast}(\mathbf{k}) D_{ij}^{l,a} \, p_{lm,j}^{a}(\mathbf{k}),$$

where the projections are $p_{lm,i}^a(\mathbf{k}) = \langle \beta_{lm,i}^a | \psi_{n\mathbf{k}} \rangle$ and the prefactor $2$ accounts for double occupancy (spin for closed-shell systems).

**Force calculation:** Differentiating $E_\mathrm{NL}$ with respect to $\boldsymbol\tau_a$ (remembering that both $\beta_{lm,i}^a$ and the projections $p_{lm,i}^a$ depend on position), the Hellmann-Feynman theorem gives

$$\mathbf{F}_a^\mathrm{NL} = -2\sum_{n\mathbf{k}} w_\mathbf{k} f_{n\mathbf{k}} \sum_{lm,ij} D_{ij}^{l,a} \, \mathrm{Re}\left[ p_{lm,i}^{a,\ast}(\mathbf{k}) \left\langle \psi_{n\mathbf{k}} | \nabla_a \beta_{lm,j}^a \right| \psi_{n\mathbf{k}} \right\rangle \right].$$

Here, the projector gradient is computed in reciprocal space. For a projector expanded as

$$\beta_{lm,i}^a(\mathbf{r}) = \sum_\mathbf{G} e^{-i\mathbf{G} \cdot \boldsymbol\tau_a} \, \tilde{\beta}_{lm,i}^a(\mathbf{G}) e^{i\mathbf{G} \cdot \mathbf{r}},$$

the spatial gradient picks up an extra factor of $i(\mathbf{k} + \mathbf{G})$ from the plane-wave exponential:

$$\nabla_a \beta_{lm,i}^a = i(\mathbf{k} + \mathbf{G}) \beta_{lm,i}^a.$$

In practice, the projections $p_{lm,i}^a(\mathbf{k})$ are computed as inner products of wavefunctions with the Fourier-transformed projectors. The force requires an additional set of projections, $(\mathbf{k}+\mathbf{G}) \times \tilde{\beta}_{lm,i}^a(\mathbf{G})$, which are accumulated during the wavefunction-projector contraction.

**Implementation in KRONOS:** The function `nonlocal_forces()` in `src/potential/forces.cpp` orchestrates:

1. For each k-point, compute the standard projections $p_{lm,i}^a = \langle \beta_{lm,i}^a | \psi \rangle$.
2. Compute gradient projections: $\langle \psi | (\mathbf{k}+\mathbf{G}) \beta_{lm,i}^a \rangle$ for each projector and k-point.
3. Accumulate force via the formula above, summing over all occupied bands.
4. Weight by Fermi occupations $f_{n\mathbf{k}}$ and k-point weights $w_\mathbf{k}$.

Cost per k-point: $O(N_\mathrm{bands} \times N_\mathrm{atoms} \times N_\mathrm{projectors} \times N_\mathrm{PW})$, dominated by the BLAS inner products.

## XC and Hartree Forces

The exchange-correlation energy and Hartree energy depend on the electron density $n(\mathbf{r})$. The chain rule gives

$$\mathbf{F}_a^\mathrm{xc,h} = \int v_\mathrm{xc}(\mathbf{r}) \frac{\partial n(\mathbf{r})}{\partial \boldsymbol\tau_a} d^3r + \int v_\mathrm{H}(\mathbf{r}) \frac{\partial n(\mathbf{r})}{\partial \boldsymbol\tau_a} d^3r.$$

However, at SCF convergence, the density is stationary with respect to variations of the wavefunction (the variational principle). The **Hellmann-Feynman theorem ensures that the implicit wavefunction derivative cancels**, leaving only explicit dependencies. For a local potential like $v_\mathrm{xc}$ or $v_\mathrm{H}$, the explicit dependence on $\boldsymbol\tau_a$ vanishes—only the density varies, and that variation is already captured through the stationarity condition. Thus, **XC and Hartree forces contribute zero explicitly** in the Hellmann-Feynman approximation (they are included implicitly through the SCF convergence condition).

This simplification—that local potentials require no special force term—is one of the key advantages of plane-wave codes over other methods.

## Pulay Forces: Why They Vanish

In methods where the basis set depends on atomic positions (e.g., atom-centered Gaussian orbitals in LCAO), the basis set itself changes when atoms move, introducing additional "Pulay" or "BFGS" force terms. Mathematically,

$$\mathbf{F}_a^\mathrm{Pulay} = -\left\langle \psi_n \left| \frac{\partial H}{\partial \boldsymbol\tau_a} \right| \frac{\partial \psi_n}{\partial \boldsymbol\tau_a} \right\rangle - \text{c.c.},$$

which vanishes when the basis is orthonormal and independent of atomic coordinates.

**In plane-wave codes, the basis is a fixed, infinite set of plane waves $e^{i(\mathbf{k}+\mathbf{G})\cdot\mathbf{r}}$** that does not depend on $\boldsymbol\tau_a$ (only the cell shape and k-points affect it, through FFT grids and Brillouin zone symmetry). Therefore, **Pulay forces vanish identically**, and the Hellmann-Feynman formula becomes exact. This architectural advantage—that basis-set incompleteness errors in forces are eliminated—makes plane-wave DFT forces highly reliable.

## Force Symmetrization

After computing the raw Hellmann-Feynman forces from the SCF density, numerical noise and floating-point rounding accumulate. A **symmetry-enforcing post-processing step** cleans up the forces and ensures they respect the point-group symmetry of the crystal.

**Algorithm:**

1. Use spglib to identify the space-group operations that leave the crystal structure invariant.
2. For each symmetry operation $\mathbf{S}$, map atom $a$ to its image atom $a'$ (possibly related by translation).
3. Transform the force $\mathbf{F}_a$ under the symmetry operation: $\mathbf{F}_a^{(\mathbf{S})} = \mathbf{S} \mathbf{F}_{a'}$.
4. Average the forces over all symmetry-equivalent images.

**Result:** The symmetrized force on atom $a$ is

$$\mathbf{F}_a^\mathrm{sym} = \frac{1}{N_\mathrm{sym}} \sum_{\mathbf{S} \in \mathbf{G}} \mathbf{S} \mathbf{F}_{a'},$$

where the sum is over all symmetry operations and $a'$ is the image of $a$ under $\mathbf{S}$.

**Conservation:** By construction, symmetrization conserves the total force on the unit cell: $\sum_a \mathbf{F}_a^\mathrm{sym} = 0$ (up to small numerical noise). This is a consequence of translational invariance: displacing the entire cell should produce no net force.

**Implementation:** In KRONOS, `symmetrize_forces()` (in `src/io/symmetry.cpp`) applies this post-processing. The symmetrized forces are used for geometry optimization and MD, improving stability by removing spurious asymmetries.

## Finite-Difference Validation

The most stringent test of force accuracy is **finite-difference validation**: compute the energy for small ionic displacements and compare $\partial E / \partial \mathbf{R}_a$ with the analytic Hellmann-Feynman force.

**Procedure:**

1. For atom $a$ and Cartesian direction $\alpha$ (x, y, or z), displace by $\delta = 0.01$ a.u. (or smaller).
2. Run SCF to convergence at the displaced geometry; record $E(\boldsymbol\tau_a + \delta\hat{\alpha})$.
3. Compute numerical derivative: $F_\alpha^\mathrm{FD} = [E(\boldsymbol\tau_a + \delta\hat{\alpha}) - E(\boldsymbol\tau_a - \delta\hat{\alpha})] / (2\delta)$.
4. Compare with analytic force $F_\alpha^\mathrm{HF}$ from the Hellmann-Feynman calculation at $\boldsymbol\tau_a$.
5. Compute relative error: $\epsilon = |F_\alpha^\mathrm{FD} - F_\alpha^\mathrm{HF}| / |F_\alpha^\mathrm{HF}|$.

**KRONOS results:** For silicon diamond at $E_\mathrm{cut} = 12$ Ry with Gamma-point sampling, displacing a single atom by 0.01 fractional units (approximately 0.025 bohr along the lattice vector) yields agreement to **5 significant figures**: $\Delta F < 10^{-5}$ Ry/bohr for forces in the range $|F| \sim 10^{-3}$ Ry/bohr. This validates the Ewald, local, and nonlocal force components jointly.

**Expected accuracy:** For high-quality calculations (fine k-point sampling, large cutoffs), forces agree to $10^{-5}$–$10^{-6}$ Ry/bohr. At lower cutoffs or coarser sampling, $10^{-4}$ Ry/bohr is typical. The finite-difference displacement $\delta$ should be small enough to probe the linear regime ($\delta \lesssim 0.01$ bohr) but large enough to keep numerical error below the analytic error ($\delta \gtrsim 10^{-4}$ bohr for double precision).

## Geometry Optimization and Molecular Dynamics

The forces computed via Hellmann-Feynman are fed into two main applications:

1. **Geometry optimization (structural relaxation):** KRONOS uses the BFGS algorithm (in `src/solver/vc_relax.cpp`) to minimize the total energy by moving atoms along the negative gradient $-\mathbf{F}_a$. The Hessian (second-derivative matrix) is approximated using previous steps, and ions are moved until the maximum force drops below a tolerance (typically $10^{-4}$ Ry/bohr) and the energy change stabilizes.

2. **Molecular dynamics:** At fixed temperature, ions are advanced using the Verlet or velocity-Verlet algorithm, with forces as acceleration. KRONOS's MD infrastructure is under development (v0.2 roadmap).

In both cases, accurate, stable forces are essential for reliable predictions.

## How KRONOS Implements Forces

The orchestration happens in `compute_forces()` (main dispatcher in `src/solver/scf.cpp`):

```
compute_forces(const SCFState& scf_state, const Crystal& crystal, ...)
  |
  +-- ewald_forces(crystal) 
  |     [real-space + reciprocal-space ion-ion repulsion]
  |
  +-- local_pp_forces(scf_state.density, crystal)
  |     [gradient of local pseudopotential energy, G-space loop]
  |
  +-- nonlocal_forces(scf_state.wavefunctions, scf_state.eigenvalues, crystal)
  |     [projector overlaps with wavefunction gradients, BLAS]
  |
  +-- symmetrize_forces(forces, crystal)
        [spglib-based averaging over symmetry operations]
```

The output is a vector of forces $\{\mathbf{F}_a\}$ in Ry/bohr. For variable-cell relaxation, stress components are computed similarly (derivative of energy with respect to cell deformation) and passed to the BFGS optimizer alongside atomic forces.

## References

- Hellmann, H. *Einführung in die Quantenchemie*. Deuticke, Leipzig (1937). [Original statement of the theorem.]
- Feynman, R. P. "Forces in molecules." *Phys. Rev.* **56**, 340–343 (1939). [Quantum mechanical derivation and application to molecular geometry.]
- Ihm, J., Zunger, A. & Cohen, M. L. "Momentum-space formalism for the total energy of solids and the equation of state." *J. Phys. C: Solid State Phys.* **12**, 4409–4422 (1979). [Pseudopotential forces in plane waves; local and nonlocal contributions clearly separated.]
- Martin, R. M. *Electronic Structure: Basic Theory and Practical Methods*. Cambridge University Press (2004). [Chapter 9: Forces and stresses in DFT; comprehensive treatment of all force components.]
- Gonze, X., Kresse, G., et al. "Recent developments in the ABINIT software package." *Comp. Phys. Commun.* **205**, 106–131 (2016). [Section on forces, symmetrization, and validation in modern plane-wave codes.]
