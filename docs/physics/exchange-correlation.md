---
title: Exchange-Correlation Functionals — LDA, GGA, and the Jacob's Ladder
description: LDA, GGA, and hybrid exchange-correlation functionals in plane-wave DFT. Perdew-Zunger LDA, PBE, PBEsol, libxc integration in KRONOS.
keywords:
  - exchange-correlation functional
  - LDA DFT
  - GGA DFT
  - PBE functional
  - Perdew-Zunger
  - libxc
  - Jacob's ladder DFT
  - density functional approximation
  - DFT XC potential
  - DFT accuracy
slug: /physics/exchange-correlation
sidebar_position: 6
---

The exchange-correlation functional $E_{\mathrm{xc}}[n]$ is the single most important quantity in Kohn-Sham density functional theory: it encapsulates all the complicated many-body electron-electron interactions beyond the simple Hartree mean-field energy. The accuracy of any DFT calculation is almost entirely determined by the quality of the XC functional chosen—choosing the right functional for the right system is the central decision in DFT calculations. KRONOS supports a hierarchy of XC approximations from the simple but robust Local Density Approximation (LDA) through generalized-gradient-corrected GGA functionals like PBE and PBEsol, as well as hybrid functionals (see `/physics/hybrid-functionals`). This page explains how XC functionals work, the progression of approximations, and practical guidance for choosing one.

## What XC Tries to Capture

In Kohn-Sham DFT, the total energy is split into kinetic, Hartree, external, and XC contributions:

$$E_{\mathrm{tot}}[n] = T_s[n] + E_H[n] + E_{\mathrm{ext}}[n] + E_{\mathrm{xc}}[n]$$

The first three terms are known exactly (or computed exactly on a grid). The XC functional is a black box that must approximate the remainder: the quantum mechanical exchange (tendency of electrons to avoid each other due to the Pauli exclusion principle) and correlation (dynamic correlation beyond exchange). This includes:

- **Self-interaction correction:** The Hartree energy falsely self-interacts an electron with itself; XC must correct this.
- **Exchange energy:** The determinantal (antisymmetric) nature of the wavefunction lowers energy compared to mean-field.
- **Correlation energy:** Beyond exchange; quantum fluctuations, ionic screening, and collective phenomena.

Because XC captures all these effects in a single functional, DFT is amazingly versatile—but the trade-off is that the exact form of $E_{\mathrm{xc}}$ is unknown. We must build approximations.

## The Jacob's Ladder of Approximations

Perdew's "Jacob's ladder" is a conceptual framework for increasingly accurate XC approximations, with each rung higher in the ladder adding more information about the electron density:

**Rung 1: LDA** — depends only on the electron density $n(\mathbf{r})$ at each point.

$$E_{\mathrm{xc}}^{\mathrm{LDA}}[n] = \int \epsilon_{\mathrm{xc}}(n(\mathbf{r})) \, n(\mathbf{r}) \, d\mathbf{r}$$

**Rung 2: GGA** — adds the gradient of the density $\nabla n(\mathbf{r})$:

$$E_{\mathrm{xc}}^{\mathrm{GGA}}[n] = \int \epsilon_{\mathrm{xc}}(n, |\nabla n|) \, n \, d\mathbf{r}$$

**Rung 3: Meta-GGA** — adds the kinetic energy density $\tau(\mathbf{r}) = \frac{1}{2} \sum_i |\nabla \psi_i(\mathbf{r})|^2$ (occupancy weighted). KRONOS does not presently implement meta-GGA.

**Rung 4: Hybrids** — mix in exact Hartree-Fock exchange. PBE0 and HSE06 are the most common examples (see `/physics/hybrid-functionals`).

**Rung 5: Double-hybrids** — incorporate MP2-like correlation on top of a hybrid. Outside the scope of KRONOS v0.1–v1.0.

Higher rungs generally improve accuracy, but also increase computational cost and can introduce instabilities. **The key insight:** each rung requires knowledge of smaller-scale physics, and for insulators and semiconductors, rung 2 (GGA) often strikes the best balance.

## LDA: Perdew-Zunger

The Local Density Approximation uses the exchange-correlation energy density of a uniform electron gas, parametrized by the local density. In Rydberg atomic units (which KRONOS uses throughout), the Perdew-Zunger LDA exchange energy is:

$$\epsilon_x^{\mathrm{LDA}}(n) = -\frac{3}{2} \left( \frac{3}{\pi} \right)^{1/3} n^{1/3} = -1.458 \, n^{1/3}$$

Correlation energy comes from a fit to quantum Monte Carlo (QMC) data for the uniform electron gas (Ceperley-Alder, 1980). The fit splits into two regimes depending on the Wigner-Seitz radius $r_s = (3 / 4\pi n)^{1/3}$:

- For $r_s < 1$ (high density): polynomial in $\log(r_s)$
- For $r_s > 1$ (low density, metallic): simpler algebraic form

KRONOS implements the full Perdew-Zunger 1981 correlation in `src/potential/xc_builtin.cpp`. LDA is fast, surprisingly accurate for ground-state structure (lattice constants within ~1%), but systematically overbinds atoms (overestimates binding energies by ~10 kcal/mol) and underestimates band gaps by ~30%.

## GGA: PBE

The Generalized Gradient Approximation extends LDA by including the gradient of the density. The PBE (Perdew-Burke-Ernzerhof) functional, published in 1996, is one of the most widely used functionals in solid-state physics. It introduces a reduced gradient:

$$s = \frac{|\nabla n|}{2 k_F n}, \quad k_F = (3\pi^2 n)^{1/3}$$

The exchange and correlation energies are then written as:

$$E_x^{\mathrm{PBE}}[n] = \int \epsilon_x^{\mathrm{LDA}}(n) \, F_x(s) \, n \, d\mathbf{r}$$

where $F_x(s)$ is an enhancement factor. Crucially, PBE satisfies the **LDA limit**: $F_x(0) = 1$, so in regions of low density gradient (or zero gradient), PBE reduces exactly to LDA. This is a key design principle that ensures stability.

PBE typically improves upon LDA:
- Lattice constants within ~0.5% (vs. LDA ~1% underbinding)
- Atomization energies improved to within ~5 kcal/mol
- Band gaps still underestimated, but less severely

PBE is "chemically accurate" for many purposes and remains the default GGA in KRONOS.

## GGA Accuracy Gains

Systematic comparisons show that GGA consistently improves over LDA for structural parameters. KRONOS validates against QE for the Si bulk test system:

- **Si lattice constant:** LDA 10.28 Bohr (underbound), PBE 10.44 Bohr (near-experimental 10.50)
- **Atomization energy:** LDA overestimates by ~15%, PBE within ~5%
- **Pressure:** LDA predicts higher bulk moduli; PBE closer to experiment

The trade-off is that GGA is slightly slower (about 20% longer wall time per SCF step due to gradient computation), but the accuracy gain justifies it for most applications.

## Spin Polarization: LSDA and Spin-GGA

For magnetic systems, KRONOS supports spin-polarized calculations where the electron density splits into majority and minority spins:

$$n(\mathbf{r}) \to \{n_\uparrow(\mathbf{r}), n_\downarrow(\mathbf{r})\}$$

The Local Spin-Density Approximation (LSDA) and its GGA extension treat each spin channel with the appropriate XC functional. Key quantities are:

- **Total density:** $n = n_\uparrow + n_\downarrow$
- **Magnetization:** $m = n_\uparrow - n_\downarrow$ (units: $\mu_B$ per atom)
- **Spin XC energy:** $E_{\mathrm{xc}}[n_\uparrow, n_\downarrow]$ evaluated from QMC fits or libxc

Activate spin polarization in KRONOS via:

```yaml
nspin: 2
```

KRONOS was validated on Fe BCC with spin-polarized PBE, recovering a magnetic moment of ~2.66 $\mu_B$ and correct ground-state energy ordering, matching Quantum ESPRESSO closely.

## libxc and KRONOS Integration

When compiled with `KRONOS_HAS_LIBXC` defined (CMake option), KRONOS dispatches XC evaluations to the **libxc** library, which implements dozens of functionals from all rungs of Jacob's ladder. This allows effortless access to PBEsol (optimized for solids), SCAN (meta-GGA), and many others.

If libxc is not available, KRONOS falls back to built-in implementations: Perdew-Zunger LDA and PBE GGA are always available. The dispatch logic lives in `src/potential/xc.cpp`:

```cpp
if (KRONOS_HAS_LIBXC && functional_name != "LDA_PZ_BUILTIN") {
  evaluate_xc_via_libxc(...);
} else {
  evaluate_xc_builtin(...);
}
```

This design ensures KRONOS is portable: it compiles and runs on systems without libxc, while offering richer functionality when libxc is present.

## Choosing a Functional

**LDA-PZ (Perdew-Zunger 1981):** The historical baseline. Overbinds atoms (~10 kcal/mol) but is very stable. Use for:
- Metals where GGA can be fragile
- Initial guesses for hybrid calculations
- Academic benchmarking

**LDA-PW (Perdew-Wang 1992):** A refinement to PZ with slightly better correlation fits. Rarely necessary if PBE is available.

**PBE (Perdew-Burke-Ernzerhof 1996):** The workhorse. Default choice for most applications:
- Semiconductors and insulators
- Metals (though LDA can be more stable in some cases)
- Molecules and solids
- Structural optimization and forces

**PBEsol (Perdew et al. 2008):** GGA reparametrized for solids by fitting to experimental lattice constants. Superior lattice constants and bulk moduli. Use for:
- High-precision structural work
- Phase stability comparisons
- When <0.5% errors in $a$ matter

**PBE0 / HSE06:** Hybrid functionals. Recover ~50% exact exchange. Dramatically improve band gaps and derivative properties, at ~4–10× computational cost. See `/physics/hybrid-functionals`.

## References

- Perdew, J. P. & Zunger, A. Self-interaction correction to density-functional approximations for many-electron systems. *Phys. Rev. B* **23**, 5048–5079 (1981).
- Perdew, J. P., Burke, K. & Ernzerhof, M. Generalized gradient approximation made simple. *Phys. Rev. Lett.* **77**, 3865–3868 (1996).
- Perdew, J. P. et al. Restoring the density-gradient expansion for exchange in solids and surfaces. *Phys. Rev. Lett.* **100**, 136406 (2008).
- Lehtola, S., Steigemann, C., Oliveira, M. J. T. & Marques, M. A. L. Recent developments in libxc — A comprehensive library of functional approximations for density functional theory. *SoftwareX* **7**, 1–5 (2018).
- Ceperley, D. M. & Alder, B. J. Ground state of the electron gas by a stochastic method. *Phys. Rev. Lett.* **45**, 566–569 (1980).
