---
title: The Stress Tensor in Plane-Wave DFT
description: Nielsen-Martin stress formalism for plane-wave DFT — kinetic, Hartree, exchange-correlation, Ewald, and pseudopotential contributions to the 3×3 stress tensor in KRONOS.
keywords:
  - DFT stress tensor
  - Nielsen-Martin stress
  - plane-wave DFT stress
  - variable cell relaxation
  - pressure DFT
  - vc-relax DFT
  - lattice optimization
  - DFT lattice parameter
  - stress tensor formalism
  - DFT equation of state
slug: /physics/stress-tensor
sidebar_position: 12
---

# The Stress Tensor in Plane-Wave DFT

The stress tensor is what you compute when the unit cell itself is a variable. It tells you which way the cell wants to deform to lower the total energy — essential for finding equilibrium lattice constants, computing equations of state, and running variable-cell molecular dynamics. In plane-wave DFT it has a particularly clean analytic form thanks to the Nielsen-Martin formalism, which derives every contribution as a strain derivative of the corresponding energy term.

## Definition and units

The stress tensor $\sigma_{\alpha\beta}$ is the negative gradient of the total energy with respect to a uniform strain tensor $u_{\alpha\beta}$ applied to the cell, divided by the cell volume:

$$\sigma_{\alpha\beta} = -\frac{1}{\Omega}\, \frac{\partial E_\mathrm{tot}}{\partial u_{\alpha\beta}}$$

Indices $\alpha, \beta \in \{x, y, z\}$. Symmetry of the stress tensor follows from conservation of angular momentum: $\sigma_{\alpha\beta} = \sigma_{\beta\alpha}$, giving 6 independent components (three diagonal, three off-diagonal). In KRONOS the natural unit is Ry/bohr³; the pressure (hydrostatic part) is

$$P = -\tfrac{1}{3}\,\mathrm{tr}(\sigma) \approx 14710.507\, \mathrm{GPa} \times P[\mathrm{Ry/bohr^3}]$$

A positive pressure means the cell wants to expand; a negative pressure means it wants to contract.

## Strain as a deformation of the cell

A uniform strain $u_{\alpha\beta}$ maps the position $\mathbf{r}$ to $\mathbf{r}' = (\mathbb{I} + u)\,\mathbf{r}$. The cell vectors $\mathbf{a}_i$ become $\mathbf{a}_i' = (\mathbb{I} + u)\,\mathbf{a}_i$, and the reciprocal vectors $\mathbf{G}$ transform as $\mathbf{G}' = (\mathbb{I} + u)^{-T}\,\mathbf{G} \approx (\mathbb{I} - u^T)\,\mathbf{G}$ to first order in $u$. The cell volume changes as $\Omega' = \Omega \det(\mathbb{I} + u) \approx \Omega (1 + \mathrm{tr}\,u)$.

Every energy contribution that depends on cell shape — and that's all of them — produces a stress term when differentiated with respect to $u$.

## Decomposition: term by term

The DFT total energy in plane-wave form is

$$E_\mathrm{tot} = E_\mathrm{kin} + E_\mathrm{H} + E_\mathrm{xc} + E_\mathrm{loc} + E_\mathrm{NL} + E_\mathrm{Ewald}$$

Each piece contributes a piece of the stress.

**Kinetic stress** — from $T_{n\mathbf{k}}(\mathbf{G}) = |\mathbf{k} + \mathbf{G}|^2$ (Rydberg) with $\mathbf{G}$ depending on strain:

$$\sigma_{\alpha\beta}^\mathrm{kin} = \frac{2}{\Omega} \sum_{n\mathbf{k}} w_\mathbf{k} f_{n\mathbf{k}} \sum_\mathbf{G} (k+G)_\alpha (k+G)_\beta\, |\psi_{n\mathbf{k}}(\mathbf{G})|^2$$

**Hartree stress** — from $\Omega^{-1}\sum_\mathbf{G} V_\mathrm{H}(\mathbf{G}) n^*(\mathbf{G})$:

$$\sigma_{\alpha\beta}^\mathrm{H} = \frac{8\pi}{\Omega} \sum_{\mathbf{G} \ne 0} \frac{|n(\mathbf{G})|^2}{G^2} \left[ \frac{2 G_\alpha G_\beta}{G^2} - \delta_{\alpha\beta} \right]$$

The negative isotropic part reflects that uniform charge density wants to expand (Coulomb repulsion).

**Exchange-correlation stress** — depends on whether the functional is LDA or GGA. For LDA only the volume-change-of-density term survives:

$$\sigma_{\alpha\beta}^\mathrm{xc,LDA} = \delta_{\alpha\beta}\, \frac{1}{\Omega}\, \int [ \epsilon_\mathrm{xc}(n) - v_\mathrm{xc}(n) ]\, n(\mathbf{r})\, d^3 r$$

For GGA, an additional term from $\nabla n$ enters; KRONOS computes this via the gradient of the density and the $v_\sigma$ derivative returned by libxc.

**Local pseudopotential stress** — from the strain derivative of $V_\mathrm{loc}^a(\mathbf{G})$ and the structure factor:

$$\sigma_{\alpha\beta}^\mathrm{loc} = \frac{1}{\Omega} \sum_a \sum_\mathbf{G} \mathrm{Re}[n^*(\mathbf{G}) e^{i\mathbf{G}\cdot\boldsymbol\tau_a}] \left[ V_\mathrm{loc}^a(\mathbf{G})\, \delta_{\alpha\beta} - 2 G_\alpha G_\beta\, \frac{d V_\mathrm{loc}^a(G)}{d(G^2)} \right]$$

The Coulomb tail of $V_\mathrm{loc}$ contributes an analytic correction (the same subtraction trick used in the local PP energy).

**Nonlocal pseudopotential stress** — Kleinman-Bylander projectors $\beta_i^a(\mathbf{k}+\mathbf{G})$ depend on $|\mathbf{k}+\mathbf{G}|$, which changes under strain. The strain derivative passes through the radial Bessel transform.

**Ewald stress** — Nielsen-Martin showed how to compute the strain derivative of the Ewald sum analytically, with both real-space and reciprocal-space pieces. The reciprocal piece has the same $G_\alpha G_\beta / G^2$ structure as the Hartree stress. KRONOS implements the closed-form expressions in `src/potential/stress.cpp::ewald_stress()`.

## Pressure and the equation of state

The hydrostatic pressure $P = -\tfrac{1}{3}\,\mathrm{tr}(\sigma)$ is what enters the enthalpy $H = E + PV$ minimized by `vc-relax` runs. At zero target pressure, vc-relax converges to a cell where $\sigma_{\alpha\beta} = 0$ for all components. For non-zero target pressure (`press_target` in YAML, in GPa), KRONOS adds $P_\mathrm{target}\, \mathbb{I}$ to the stress and minimizes the enthalpy:

$$\sigma_{\alpha\beta}^\mathrm{eff} = \sigma_{\alpha\beta} + P_\mathrm{target}\,\delta_{\alpha\beta}$$

Driving this to zero gives the cell at the target external pressure.

## Anisotropic stress and crystal symmetry

The full $3\times 3$ tensor reveals shape-changing forces missed by a scalar pressure. A cubic crystal at equilibrium has $\sigma_{\alpha\beta} = \frac{P}{3}\,\delta_{\alpha\beta}$ purely. A tetragonal crystal forced into a cubic cell would show $\sigma_{xx} = \sigma_{yy} \ne \sigma_{zz}$. Off-diagonal $\sigma_{\alpha\beta}$ would force a monoclinic distortion. KRONOS preserves space-group symmetry by symmetrizing the stress tensor with the same spglib operations used for forces:

$$\sigma^\mathrm{sym}_{\alpha\beta} = \frac{1}{|G|} \sum_{R \in G} R_{\alpha\gamma} R_{\beta\delta}\, \sigma_{\gamma\delta}$$

over the point group $G$ of the cell. This eliminates symmetry-breaking floating-point noise and ensures vc-relax converges along symmetry-preserving paths.

## Variable-cell relaxation in KRONOS

`vc-relax` optimizes both atomic positions and cell vectors simultaneously, using a $(3N + 6)$-dimensional BFGS quasi-Newton step (3 atomic-position dimensions plus 6 unique strain components). At each step:

1. Run SCF to convergence on the current cell.
2. Compute forces and the full stress tensor.
3. Symmetrize both.
4. Compute the enthalpy gradient and BFGS step.
5. Apply the strain to the cell; move atoms.
6. Repeat until $\max|\mathbf{F}_a| < \mathrm{tol}_\mathrm{F}$ AND $\max|\sigma_{\alpha\beta} + P_\mathrm{target}\,\delta_{\alpha\beta}| < \mathrm{tol}_\sigma$.

Implementation: `src/solver/vc_relax.cpp`.

## How KRONOS implements this

The stress calculation lives in `src/potential/stress.cpp`. Each contribution (kinetic, Hartree, XC, local PP, nonlocal PP, Ewald) is computed by a dedicated function and accumulated into the 6-component Voigt-style tensor. The total stress is exposed via `StressResult::tensor()` (3×3 matrix) and `StressResult::pressure_gpa()` (scalar).

KRONOS has unit-test coverage for each component on a Si bulk cell, asserting agreement with QE stress to within $10^{-5}\,\mathrm{Ry/bohr^3}$.

## References

- Nielsen, O. H. & Martin, R. M. "Quantum-mechanical theory of stress and force", *Phys. Rev. B* **32**, 3780 (1985).
- Nielsen, O. H. & Martin, R. M. "Stresses in semiconductors: Ab initio calculations on Si, Ge, and GaAs", *Phys. Rev. B* **32**, 3792 (1985).
- Dal Corso, A. "Density-functional perturbation theory with ultrasoft pseudopotentials", *Phys. Rev. B* **64**, 235118 (2001) — stress with USPP/PAW.
- Wentzcovitch, R. M. "Invariant molecular-dynamics approach to structural phase transitions", *Phys. Rev. B* **44**, 2358 (1991) — variable-cell MD.
- Martin, R. M. *Electronic Structure*, Ch. 9.
