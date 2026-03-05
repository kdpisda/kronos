# KRONOS Architecture Guide

## Overview

KRONOS (Kohn-Residual Optimized Numerics Over Silicon) is a plane-wave pseudopotential Density Functional Theory engine. It solves the Kohn-Sham equations self-consistently to find ground-state electronic structure of periodic crystalline systems.

All quantities use **Rydberg atomic units**: energies in Ry, lengths in bohr, wavefunctions in the plane-wave (G-space) representation.

---

## SCF Loop: The Central Algorithm

The self-consistent field (SCF) loop is KRONOS's core computation. Here is the numbered step-by-step flow:

```
┌─────────────────────────────────────────────────┐
│  1. Parse YAML input + load UPF pseudopotentials │
│  2. Build PlaneWaveBasis (G-vectors ≤ ecutwfc)   │
│  3. Build FFTGrid (sized for ecutrho ≥ 4×ecutwfc)│
│  4. Generate k-points (Monkhorst-Pack + symmetry)│
│  5. Initialize density: superposition of atomic ρ│
├─────────────────────────────────────────────────┤
│  6. SCF ITERATION LOOP:                          │
│     a. ρ(r) → ρ(G)         [FFT forward]        │
│     b. V_H(G) = 4π ρ(G)/|G|²  [Hartree]        │
│     c. V_xc(r) from ρ(r)   [libxc / built-in]   │
│     d. V_loc(G) from pseudopotentials            │
│     e. V_eff = V_H + V_xc + V_loc               │
│     f. For each k-point:                         │
│        - Build H = T + V_eff + V_NL              │
│        - Davidson eigensolver → {ε_nk, ψ_nk}    │
│     g. Fermi level by bisection → occupations f  │
│     h. New density: ρ(r) = Σ f_nk |ψ_nk(r)|²   │
│     i. Check convergence: |ΔE| and |Δρ|         │
│     j. Pulay/DIIS density mixing                 │
├─────────────────────────────────────────────────┤
│  7. Post-convergence:                            │
│     - Hellmann-Feynman forces                    │
│     - Total energy decomposition                 │
│     - Band structure / DOS (if requested)        │
│  8. Output: JSON summary + HDF5 files            │
└─────────────────────────────────────────────────┘
```

---

## Component Interaction Diagram

```
Input (YAML)
    │
    ▼
┌──────────┐     ┌──────────────┐     ┌──────────┐
│  Crystal  │────▶│PlaneWaveBasis│────▶│  FFTGrid │
│ (lattice, │     │ (G-vectors)  │     │ (r↔G)    │
│  atoms)   │     └──────┬───────┘     └────┬─────┘
└──────────┘            │                   │
                        ▼                   ▼
              ┌─────────────────────────────────┐
              │        Potentials                │
              │  Hartree  XC  LocalPP  NonlocalPP│
              │           Ewald                  │
              └─────────────┬───────────────────┘
                            ▼
              ┌─────────────────────────────────┐
              │        Hamiltonian               │
              │   H|ψ⟩ = T|ψ⟩ + V_eff·ψ + V_NL │
              └─────────────┬───────────────────┘
                            ▼
              ┌─────────────────────────────────┐
              │     Davidson Eigensolver         │
              │  → eigenvalues ε_nk              │
              │  → wavefunctions ψ_nk            │
              └─────────────┬───────────────────┘
                            ▼
              ┌─────────────────────────────────┐
              │     FermiSolver + Mixing         │
              │  → occupations f_nk              │
              │  → new density ρ(r)              │
              └─────────────┬───────────────────┘
                            ▼
              ┌─────────────────────────────────┐
              │         SCFSolver                │
              │  Orchestrates the full loop      │
              │  → SCFResult                     │
              └─────────────┬───────────────────┘
                            ▼
              ┌─────────────────────────────────┐
              │    PostProcessing                │
              │  BandStructure / DOS / Forces    │
              └─────────────────────────────────┘
                            ▼
              Output (JSON + HDF5)
```

---

## Key Algorithms

### Davidson Eigensolver

The Davidson iterative diagonalization finds the lowest N eigenvalues of H without forming the full matrix. It works in a subspace of dimension up to 3×N_bands:

1. Start with initial guess vectors (random or from previous step)
2. Apply H to each vector: H|ψ_i⟩
3. Build reduced Hamiltonian: H_ij = ⟨ψ_i|H|ψ_j⟩
4. Diagonalize the small matrix (LAPACK dsyev)
5. Compute residuals: r_i = (H - ε_i)|ψ_i⟩
6. If all residuals < tolerance: converged
7. Apply preconditioner (kinetic energy diagonal): t_i = r_i / (ε_i - T_G)
8. Expand subspace with new vectors; orthogonalize
9. Repeat from step 2

Falls back to LOBPCG if residuals diverge (> 1e3).

### Pulay/DIIS Density Mixing

Minimizes the residual ‖R‖ = ‖ρ_out - ρ_in‖ using a linear combination of the last M density/residual pairs:

1. Store history: {ρ_in^(i), R^(i) = ρ_out^(i) - ρ_in^(i)} for i = 1..M
2. Build overlap matrix: B_ij = ⟨R^(i)|R^(j)⟩
3. Solve constrained optimization: min ‖Σ c_i R^(i)‖² subject to Σ c_i = 1
4. New density: ρ_new = Σ c_i [ρ_in^(i) + α·R^(i)]

History depth M = 8 by default. First step uses simple linear mixing.

### Ewald Summation

Computes the electrostatic energy of a periodic array of point charges by splitting into fast-converging real-space and reciprocal-space sums:

```
E_ion = E_real + E_recip + E_self + E_charged

E_real  = Σ'_{T,i<j} Z_i Z_j erfc(η|r_ij + T|) / |r_ij + T|
E_recip = (4π/Ω) Σ_{G≠0} |S(G)|² exp(-|G|²/4η²) / |G|²
E_self  = -η/√π Σ_i Z_i²
```

The parameter η is chosen to balance convergence: η = √π (N/Ω²)^(1/6).

### Kleinman-Bylander (KB) Nonlocal PP

The nonlocal pseudopotential in separable KB form:

```
V_NL|ψ⟩ = Σ_{I,lm} D_l |β_{Ilm}⟩⟨β_{Ilm}|ψ⟩
```

Projections ⟨β|ψ⟩ are computed via GEMM for all projectors simultaneously. The β projectors are precomputed in G-space from the radial functions β_l(r) and spherical harmonics.

### Fermi Level Bisection

For metals (smearing enabled), find E_F such that:

```
Σ_{nk} w_k · f(ε_nk, E_F, σ) = N_electrons
```

Uses bisection on E_F. Smearing functions: Gaussian, Marzari-Vanderbilt (cold smearing), Fermi-Dirac.

---

## Data Flow

```
YAML input
  → Crystal (lattice vectors in Å → bohr, atoms in fractional coords)
    → PlaneWaveBasis (G-vectors where |G|²/2 ≤ ecutwfc)
      → FFTGrid (r↔G transforms, grid sized for ecutrho)
        → Potentials (V_H, V_xc, V_loc in G-space; V_eff on r-grid)
          → Hamiltonian (H|ψ⟩ application via FFT + GEMM)
            → Davidson (eigenvalues + wavefunctions)
              → FermiSolver (occupations)
                → New density (ρ from |ψ|²)
                  → Convergence check → loop or exit
```

---

## Type System

| Type | Definition | Usage |
|------|-----------|-------|
| `real_t` | `double` | Scalar quantities |
| `complex_t` | `std::complex<double>` | Wavefunction coefficients |
| `Vec3` | `std::array<double, 3>` | Positions, k-vectors, forces |
| `Mat3` | `std::array<std::array<double,3>,3>` | Lattice, reciprocal lattice |
| `CVec` | `std::vector<complex_t>` | G-space wavefunctions/potentials |
| `RVec` | `std::vector<double>` | Real-space densities/potentials |

---

## Units Convention

KRONOS uses **Rydberg atomic units** throughout:

| Quantity | Unit | Conversion |
|----------|------|------------|
| Energy | Ry | 1 Ry = 13.6057 eV |
| Length | bohr | 1 bohr = 0.529177 Å |
| Force | Ry/bohr | |
| Wavefunction cutoff | Ry | |
| Temperature | Ry (via k_B) | |

Input lattice vectors are in **Angstrom** and converted to bohr internally. Atom positions are in **fractional coordinates**.

---

## Source Directory Map

```
src/
├── core/
│   ├── types.hpp          # Fundamental types: Vec3, Mat3, CVec, RVec, enums
│   ├── constants.hpp      # Physical constants (CODATA 2018) and conversions
│   ├── crystal.hpp/cpp    # Crystal class: lattice, atoms, volume, coord transforms
│   └── element_data.hpp   # Periodic table data (Z, mass, symbol)
│
├── basis/
│   ├── plane_wave.hpp/cpp # PlaneWaveBasis: G-vector enumeration, kinetic energies
│   └── fft_grid.hpp/cpp   # FFTGrid: FFTW3 wrapper, scatter/gather G↔grid
│
├── io/
│   ├── input_parser.hpp/cpp  # YAML input parser (strict schema)
│   ├── upf_parser.hpp/cpp    # UPF pseudopotential reader + validator
│   └── output_writer.hpp/cpp # JSON summary + HDF5 output
│
├── potential/
│   ├── hartree.hpp/cpp       # Poisson solver: V_H(G) = 4πρ(G)/|G|²
│   ├── xc.hpp/cpp            # XC evaluator (libxc wrapper + built-in LDA)
│   ├── local_pp.hpp/cpp      # Local pseudopotential V_loc(G)
│   ├── nonlocal_pp.hpp/cpp   # KB projectors, V_NL|ψ⟩ via GEMM
│   ├── ewald.hpp/cpp         # Ewald ion-ion energy and forces
│   ├── gga_gradient.hpp/cpp  # |∇ρ|² for GGA functionals
│   └── forces.hpp/cpp        # Hellmann-Feynman force calculator
│
├── solver/
│   ├── scf.hpp/cpp           # SCF loop orchestrator → SCFResult
│   ├── davidson.hpp/cpp      # Davidson iterative eigensolver
│   ├── mixing.hpp/cpp        # Linear, Pulay/DIIS, Kerker mixing
│   ├── fermi.hpp/cpp         # Fermi level finder (bisection)
│   └── bfgs.hpp/cpp          # BFGS geometry optimizer
│
├── hamiltonian/
│   └── hamiltonian.hpp/cpp   # H|ψ⟩ = T|ψ⟩ + V_eff·ψ + V_NL|ψ⟩
│
├── postprocessing/
│   ├── band_structure.hpp/cpp # Band structure along k-path
│   └── dos.hpp/cpp            # Density of states calculator
│
├── gpu/
│   ├── fft.hpp               # GPU FFT abstraction (cuFFT/rocFFT stubs)
│   ├── blas.hpp              # GPU BLAS abstraction (cuBLAS/rocBLAS stubs)
│   └── memory.hpp            # GPU memory management
│
├── utils/
│   ├── timer.hpp/cpp         # Scoped timer + profiling registry
│   └── logger.hpp/cpp        # Structured JSON logger (stderr)
│
└── main.cpp                  # Entry point: parse args, run calculation
```
