---
title: SCF Workflow Flowchart
description: Top-to-bottom flow of the KRONOS self-consistent field loop — parse input, build basis, initialize density, iterate to convergence, output forces and stress.
keywords:
  - SCF workflow
  - self-consistent field DFT
  - plane-wave DFT loop
  - Kohn-Sham iteration
  - Hartree XC potential
  - DIIS density mixing
  - Davidson eigensolver
  - KRONOS
slug: /architecture/scf-flowchart
sidebar_position: 2
---

# SCF Workflow Flowchart

The self-consistent field loop is the central algorithm in KRONOS. The outer flow is linear — parse input, build the plane-wave basis, initialize the electron density, iterate until convergence, then post-process — while the inner SCF loop iterates between the Hamiltonian application, eigensolver, Fermi level solver, and density mixer until energy and density residuals fall below the convergence thresholds. See [Key Algorithms](algorithms.md) for pseudocode details on each sub-step, and [Data Flow](data-flow.md) for how the key data objects move between modules.

The self-consistent field loop is the central algorithm. The outer flow is
linear (parse, build, iterate, post-process), while the SCF loop iterates
until convergence.

```mermaid
flowchart TD
    A1[1. Parse YAML input<br/>strict schema, unknown keys abort] --> A2[2. Load UPF pseudopotentials<br/>validate norm conservation]
    A2 --> A3[3. Generate k-points<br/>Monkhorst-Pack + time-reversal folding<br/>compute k_max = max k_cart]
    A3 --> A4[4. Build PlaneWaveBasis<br/>enumerate G with k+G² ≤ ecutwfc<br/>expanded sphere via k_max]
    A4 --> A5[5. Build FFTGrid<br/>dims from ecutrho ≥ 4·ecutwfc<br/>FFTW3 forward/inverse plans]
    A5 --> A6[6. Pre-compute static quantities<br/>V_loc G  ·  G²  ·  Nonlocal PP D_ij]
    A6 --> A7[7. Initialize density<br/>superposition of atomic ρ r<br/>uniform fallback]

    A7 --> SCF

    subgraph SCF [SCF Iteration Loop]
        direction TB
        L1[a. ρ r → ρ G  via FFT] --> L2[b. V_H G = 8π·ρ G / G²<br/>Hartree / Poisson]
        L2 --> L3[c. V_xc r from ρ r<br/>libxc / built-in<br/>GGA: also vsigma]
        L3 --> L4[d. V_eff G = V_H G + V_loc G<br/>V_eff r = IFFT V_eff G + V_xc r]
        L4 --> L5[e. For each k-point:<br/>mask ψ to active PW set<br/>H ψ = T+V_eff+V_NL<br/>Davidson → ε_nk, ψ_nk]
        L5 --> L6[f. Fermi bisection → f_nk]
        L6 --> L7[g. New density<br/>ρ r = Σ_nk w_k·f_nk· ψ_nk r ²]
        L7 --> L8[h. Total energy<br/>E_band − E_H + E_xc − ∫V_xc·n + E_smear]
        L8 --> L9{i. Converged?<br/>dE  < ε_E AND<br/>dn_G  < ε_n}
        L9 -- "no" --> L10[j. Pulay/DIIS density mix<br/>Kerker for metals<br/>clamp ρ ≥ 0, renormalize]
        L10 --> L1
    end

    L9 -- "yes" --> P1[8. Post-SCF<br/>Ewald ion-ion energy<br/>Hellmann-Feynman forces<br/>spglib force symmetrization]
    P1 --> P2[9. Output<br/>JSON summary, HDF5<br/>atomic write via temp+rename]
```
