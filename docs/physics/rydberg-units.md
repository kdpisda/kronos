---
title: Rydberg Atomic Units in Plane-Wave DFT
description: Rydberg vs Hartree atomic units in plane-wave DFT — definitions, conversions, kinetic operator forms, why KRONOS uses Rydberg.
keywords:
  - Rydberg atomic units
  - Hartree atomic units
  - atomic units conversion
  - DFT units
  - plane-wave DFT
  - Quantum ESPRESSO units
  - bohr
  - Ry to eV
  - kinetic energy DFT
  - stress GPa conversion
slug: /physics/rydberg-units
sidebar_position: 1
---

# Rydberg Atomic Units in Plane-Wave DFT

All equations in KRONOS are written in **Rydberg atomic units** — the same convention
used by Quantum ESPRESSO and most plane-wave codes in the pseudopotential tradition.
This page defines both flavors of atomic units, explains where the factor-of-2
difference comes from, lists every conversion you will need, and shows the exact
operator forms that appear in the code. If you have ever debugged a kinetic energy
that was exactly half what it should be, this page is for you.

## Atomic units: the natural scale

Ordinary SI units are inconvenient for atomic-scale calculations. Energies of interest
are ~10⁻¹⁸ J; electron–electron distances are ~10⁻¹⁰ m. Working in SI forces every
equation to carry large powers of 10, and it hides the physics behind conversion
constants.

Atomic units remove this clutter by choosing the four fundamental constants of
one-electron quantum mechanics as the unit basis:

| Constant | Symbol | Role |
|----------|--------|------|
| Electron mass | $m_e$ | mass unit |
| Elementary charge | $e$ | charge unit |
| Reduced Planck constant | $\hbar$ | action unit |
| Bohr radius | $a_0 = \hbar^2/(m_e e^2)$ | length unit |

In any atomic unit system these constants are each set to a small integer (1 or 2).
The choice of which integer determines which *flavor* of atomic units you get.

## Hartree vs Rydberg: the factor of two

Two conventions dominate the DFT literature. They share the same length unit (the
bohr) but differ in the energy scale.

**Hartree atomic units** (used by VASP, CP2K, Gaussian) set:

$$\hbar = 1,\quad m_e = 1,\quad e^2 = 1,\quad a_0 = 1$$

With these definitions the Schrödinger kinetic operator for a free particle becomes:

$$\hat{T}_\mathrm{Ha} = -\nabla^2 / 2$$

and the ground-state energy of hydrogen is $-1/2\ \mathrm{Ha}$.

**Rydberg atomic units** (used by Quantum ESPRESSO, KRONOS, many pseudopotential codes) set:

$$\hbar = 1,\quad m_e = \tfrac{1}{2},\quad e^2 = 2,\quad a_0 = 1$$

Halving $m_e$ and doubling $e^2$ leaves the Bohr radius $a_0 = \hbar^2/(m_e e^2)$
unchanged — so lengths are identical in both systems. But the energy unit doubles:

$$E_\mathrm{Ry} = \frac{m_e e^4}{2\hbar^2}\Bigg|_\mathrm{Ry\ defs} = \frac{(\tfrac{1}{2})(2)^2}{2 \cdot 1^2} = 1\ \mathrm{Ry}$$

which equals $\tfrac{1}{2}$ Hartree. The kinetic operator becomes:

$$\hat{T}_\mathrm{Ry} = -\nabla^2$$

(the $m_e = 1/2$ absorbs the factor of 2 that would otherwise sit in the denominator).

**Summary of the factor-of-two origin:** Rydberg units halve the electron mass. This
removes the $1/2$ from the kinetic operator and doubles the Hartree potential prefactor
because $e^2 = 2$ in Rydberg units. Every difference between Hartree and Rydberg
operator forms traces back to these two substitutions.

## Conversion table

| Quantity | Rydberg AU | Hartree AU | SI / conventional |
|----------|-----------|------------|-------------------|
| Energy | $1\ \mathrm{Ry}$ | $\tfrac{1}{2}\ \mathrm{Ha}$ | $13.6057\ \mathrm{eV}$ |
| Energy | $2\ \mathrm{Ry}$ | $1\ \mathrm{Ha}$ | $27.2114\ \mathrm{eV}$ |
| Length | $1\ \mathrm{bohr}$ | $1\ \mathrm{bohr}$ | $0.529177\ \mathrm{Å}$ |
| Force | $1\ \mathrm{Ry/bohr}$ | $\tfrac{1}{2}\ \mathrm{Ha/bohr}$ | $25.7110\ \mathrm{nN}$ |
| Pressure | $1\ \mathrm{Ry/bohr^3}$ | $\tfrac{1}{2}\ \mathrm{Ha/bohr^3}$ | $14710.507\ \mathrm{GPa}$ |
| Velocity (time unit: $\hbar/\mathrm{Ry}$) | $1\ \mathrm{bohr \cdot Ry}/\hbar$ | — | $1.09 \times 10^6\ \mathrm{m/s}$ |
| $m_e$ | $1/2$ | $1$ | $9.10938 \times 10^{-31}\ \mathrm{kg}$ |
| $e^2$ | $2$ | $1$ | $1.43996\ \mathrm{eV \cdot nm}$ |

The energy conversion $1\ \mathrm{Ha} = 2\ \mathrm{Ry}$ is the single most important
number to memorize. Force and pressure conversions follow immediately because the
length unit is shared.

## Why KRONOS uses Rydberg

KRONOS targets compatibility with Quantum ESPRESSO (QE) benchmarks as its primary
validation reference. QE, PWscf, and most pseudopotential codes in the norm-conserving
and PAW traditions were written in Rydberg units because the earliest plane-wave codes
(notably those of Arias, Payne, and Gonze) followed the convention of Hamann,
Schlüter, and Chiang (1979) for norm-conserving pseudopotentials, which uses Rydberg
units.

Concretely, UPF pseudopotential files (the universal format used by QE and KRONOS)
store local potentials in Rydberg units. Reading a UPF file and immediately comparing
energies against a Hartree-unit code requires multiplying by 2. KRONOS avoids this
entire class of off-by-two bugs by staying in Rydberg throughout:

- UPF files read in → no conversion needed
- QE reference energies → direct comparison, no factor of 2
- All KRONOS test baselines are in Ry, matching QE output directly

If you are porting formulas from a Hartree-unit paper or code, multiply every
energy by $\tfrac{1}{2}$ and every potential (energy/charge) by $\tfrac{1}{2}$ before
using it in KRONOS.

## Operator forms in Rydberg units

The following table shows the three operators where Hartree and Rydberg units differ
most consequentially. These are the forms implemented in `src/hamiltonian/` and
`src/potential/`.

| Operator | Hartree atomic units | Rydberg atomic units |
|----------|---------------------|---------------------|
| Kinetic (G-space) | $\hat{T} \psi_\mathbf{G} = \tfrac{1}{2}\|\mathbf{k}+\mathbf{G}\|^2 \psi_\mathbf{G}$ | $\hat{T} \psi_\mathbf{G} = \|\mathbf{k}+\mathbf{G}\|^2 \psi_\mathbf{G}$ |
| Hartree potential (G-space) | $V_H(\mathbf{G}) = \dfrac{4\pi\, n(\mathbf{G})}{\|\mathbf{G}\|^2}$ | $V_H(\mathbf{G}) = \dfrac{8\pi\, n(\mathbf{G})}{\|\mathbf{G}\|^2}$ |
| Bare Coulomb | $V(\mathbf{r}) \to -Z/r$ | $V(\mathbf{r}) \to -2Z/r$ |

### Kinetic energy

In reciprocal space (plane-wave basis) the kinetic operator is diagonal. For a
wavefunction expanded as $\psi_{n\mathbf{k}}(\mathbf{r}) = \sum_\mathbf{G} c_{n\mathbf{k}\mathbf{G}} e^{i(\mathbf{k}+\mathbf{G})\cdot\mathbf{r}}$:

$$\hat{T}\psi_{n\mathbf{k}} = \sum_\mathbf{G} |\mathbf{k}+\mathbf{G}|^2\, c_{n\mathbf{k}\mathbf{G}}\, e^{i(\mathbf{k}+\mathbf{G})\cdot\mathbf{r}}$$

There is no $1/2$ prefactor. The plane-wave cutoff is therefore `ecutwfc` in Ry,
defined as $|\mathbf{k}+\mathbf{G}|^2 \leq E_\mathrm{cut}$ — not
$|\mathbf{k}+\mathbf{G}|^2/2 \leq E_\mathrm{cut}$.

### Hartree potential

The Poisson equation in G-space, with $e^2 = 2$ in Rydberg units:

$$V_H(\mathbf{G}) = \frac{8\pi\, n(\mathbf{G})}{|\mathbf{G}|^2}$$

The $\mathbf{G} = \mathbf{0}$ component (average electrostatic) is set to zero for
charge-neutral periodic systems (the uniform background cancels it).

### Local pseudopotential Fourier transform

The local part of the pseudopotential has a Coulomb tail $V_\mathrm{loc}(r) \to -2Z/r$
as $r \to \infty$ (Rydberg convention). In G-space this tail contributes an analytic
term:

$$V_\mathrm{loc}(\mathbf{G}) = V_\mathrm{loc}^\mathrm{short}(\mathbf{G}) + \frac{8\pi Z}{\Omega |\mathbf{G}|^2}$$

where $\Omega$ is the unit cell volume. The short-range part is the smooth remainder
after subtracting the Coulomb tail in real space.

## Stress and pressure

The stress tensor $\sigma_{\alpha\beta}$ in KRONOS is computed in units of
$\mathrm{Ry/bohr^3}$. To convert to GPa:

$$P\ [\mathrm{GPa}] = \sigma\ [\mathrm{Ry/bohr^3}] \times 14710.507\ \mathrm{GPa\cdot bohr^3/Ry}$$

The conversion factor follows from:

$$1\ \mathrm{Ry} = 2.17987 \times 10^{-18}\ \mathrm{J}, \quad 1\ \mathrm{bohr} = 5.29177 \times 10^{-11}\ \mathrm{m}$$

$$1\ \mathrm{Ry/bohr^3} = \frac{2.17987 \times 10^{-18}}{(5.29177 \times 10^{-11})^3}\ \mathrm{Pa} \approx 1.4711 \times 10^{13}\ \mathrm{Pa} = 14710.5\ \mathrm{GPa}$$

Pressure is reported as the negative one-third trace of the stress:

$$P = -\frac{1}{3}\,\mathrm{tr}(\sigma)$$

(positive pressure = compression, following the geophysics/thermodynamics sign convention
used by QE and most DFT codes).

## Common pitfalls

**Reading Hartree-unit output and forgetting the factor of 2.** If a VASP
or CP2K calculation reports a total energy of $-4.12\ \mathrm{Ha}$, the equivalent
KRONOS value is $-8.24\ \mathrm{Ry}$. The numbers look wildly different even
though they represent the same physical energy. Always check the unit label in output files.

**ecutwfc in Hartree vs Rydberg.** The cutoff $E_\mathrm{cut} = 40\ \mathrm{Ry}$ is
equivalent to $20\ \mathrm{Ha}$. VASP `ENCUT` is in eV ($40\ \mathrm{Ry} \approx
544\ \mathrm{eV}$). Getting this wrong produces a basis set that is either half the size
or twice the size intended, with order-of-magnitude errors in convergence.

**Applying the 1/2 kinetic prefactor from a Hartree paper.** If you copy the kinetic
energy formula from a textbook written in Hartree units and add a $1/2$, your kinetic
energy will be half the correct value. KRONOS has caught this class of bug repeatedly —
see the Rydberg call-out at the top of `docs/physics_notes.md`.

**ecutrho must be at least 4× ecutwfc (norm-conserving) or 12× (PAW).** These
factors are the same in both unit systems because they relate two Rydberg cutoffs.
No unit conversion is needed here, but they are easy to violate when translating
input files from Hartree-unit codes that express the charge density cutoff differently.

**Force sum rule in mixed-unit codes.** If forces from one layer of a code are
accumulated in Ry/bohr but a second layer computes them in Ha/bohr, the force sum
(Newton's third law self-consistency) will appear violated by a factor of 2. The
KRONOS force validation tests (`test_forces`) catch this by comparing to finite
differences.

## References

- Martin, R. M. *Electronic Structure: Basic Theory and Practical Methods*, Cambridge University Press, 2004 — Appendix on units
- Quantum ESPRESSO documentation — Units conventions
- NIST CODATA recommended values (2018): [physics.nist.gov/cuu/Constants](https://physics.nist.gov/cuu/Constants)
- Hamann, D. R., Schlüter, M. & Chiang, C. (1979). *Norm-Conserving Pseudopotentials*, Phys. Rev. Lett. **43**, 1494 — original NCP paper establishing the Rydberg convention
