---
title: Unit Conventions (Rydberg Atomic Units)
description: KRONOS uses Rydberg atomic units throughout — energies in Ry, lengths in bohr — which differ from Hartree atomic units by factors of 2 in key formulas.
keywords:
  - Rydberg atomic units
  - DFT units
  - Hartree vs Rydberg
  - plane-wave DFT units
  - kinetic energy Rydberg
  - Hartree potential formula
  - KRONOS unit system
slug: /architecture/unit-conventions
sidebar_position: 6
---

# Unit Conventions (Rydberg Atomic Units)

KRONOS uses Rydberg atomic units throughout the codebase. This is the same convention used by Quantum ESPRESSO and differs from Hartree atomic units — used by VASP and CP2K — by factors of 2 in the kinetic energy, Hartree potential, and Coulomb tail. Mixing up conventions is the most common source of silent numerical errors when porting formulas from papers; this page lists every critical formula with its Rydberg form. See [Key Algorithms](algorithms.md) for how these formulas appear in the solver pseudocode.

KRONOS uses **Rydberg atomic units** throughout. This differs from Hartree
atomic units by factors of 2 in key places.

| Quantity | Rydberg AU | Hartree AU | Conversion |
|----------|-----------|------------|------------|
| Energy unit | 1 Ry = 13.6057 eV | 1 Ha = 27.2114 eV | 1 Ha = 2 Ry |
| Length unit | 1 bohr = 0.529177 A | same | -- |
| Kinetic energy | T = \|k+G\|^2 | T = \|k+G\|^2 / 2 | factor of 2 |
| Hartree potential | V_H(G) = 8pi n(G)/\|G\|^2 | V_H(G) = 4pi n(G)/\|G\|^2 | factor of 2 |
| Coulomb potential | V = -2Z/r | V = -Z/r | factor of 2 |
| Force unit | Ry/bohr | Ha/bohr | factor of 2 |

Input lattice vectors are specified in **Angstrom** and converted to bohr
internally. Atom positions are in **fractional coordinates** (dimensionless,
range [0,1)). The wavefunction cutoff `ecutwfc` and density cutoff `ecutrho`
are in Ry.

### Critical Rydberg-Unit Formulas

```
Kinetic:       T|psi>_G  = |k+G|^2 * psi_G           (NOT |k+G|^2/2)

Hartree:       V_H(G)    = 8*pi * n(G) / |G|^2       (NOT 4*pi)

Coulomb tail:  V_loc(r) --> -2*Z/r  as r --> inf      (NOT -Z/r)

Local PP FT:   V_loc(G)  = V_loc_short(G) + 8*pi*Z / (Omega * |G|^2)
               where the analytic term subtracts the -2Z/r Coulomb tail

Total energy:  E_tot = E_band - E_H + E_xc - int(V_xc * n) + E_ewald + E_smearing
               (double-counting correction removes Hartree and XC overcounting)
```

### Type System

| Type | Definition | Usage |
|------|-----------|-------|
| `real_t` | `double` | Scalar quantities |
| `complex_t` | `std::complex<double>` | Wavefunction coefficients (always float64) |
| `Vec3` | `std::array<double, 3>` | Positions, k-vectors, G-vectors, forces |
| `Mat3` | `std::array<std::array<double,3>,3>` | Lattice, reciprocal lattice |
| `CVec` | `std::vector<complex_t>` | G-space wavefunctions, potentials |
| `RVec` | `std::vector<double>` | Real-space densities, potentials |
