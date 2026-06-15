# KRONOS Architecture Guide

## 1. High-Level Overview

KRONOS (Kohn-Residual Optimized Numerics Over Silicon) is a research-grade,
ab initio plane-wave pseudopotential Density Functional Theory (DFT) engine.
It solves the Kohn-Sham equations self-consistently to compute:

- Ground-state total energy and its decomposition (kinetic, Hartree, XC,
  local PP, nonlocal PP, Ewald ion-ion)
- Kohn-Sham eigenvalues and band structure
- Electron density on real-space and reciprocal-space grids
- Hellmann-Feynman ionic forces
- Density of states

KRONOS targets norm-conserving pseudopotentials (NCPP) with LDA and GGA
exchange-correlation functionals. It reads standard UPF v2 pseudopotential
files and YAML input, and writes JSON summaries and HDF5 binary output.

Within the DFT ecosystem, KRONOS occupies the same niche as Quantum ESPRESSO's
PWscf module -- a plane-wave code operating in reciprocal space with periodic
boundary conditions. It is designed for periodic crystalline systems (bulk
solids, surfaces, 2D materials with vacuum padding) using the pseudopotential
approximation to replace core electrons.

All internal quantities use **Rydberg atomic units** (energies in Ry, lengths
in bohr). The code is written in C++20 with a GPU abstraction layer for
future CUDA/HIP offloading.

---

## 2. SCF Flowchart

The self-consistent field loop is the central algorithm. The outer flow is
linear (parse, build, iterate, post-process), while the SCF loop iterates
until convergence.

```
 +===================================================================+
 |                       KRONOS SCF WORKFLOW                         |
 +===================================================================+
 |                                                                   |
 |  [1] Parse YAML input file (strict schema, unknown keys abort)   |
 |      |                                                            |
 |      v                                                            |
 |  [2] Load UPF pseudopotentials (validate norm conservation)      |
 |      |                                                            |
 |      v                                                            |
 |  [3] Generate k-points (Monkhorst-Pack + time-reversal folding)  |
 |      Compute k_max = max|k_cart| over all k-points               |
 |      |                                                            |
 |      v                                                            |
 |  [4] Build PlaneWaveBasis                                        |
 |      Enumerate G-vectors where |k+G|^2 <= ecutwfc for any k      |
 |      (expanded sphere using k_max)                                |
 |      |                                                            |
 |      v                                                            |
 |  [5] Build FFTGrid                                               |
 |      Grid dimensions from ecutrho >= 4 * ecutwfc                 |
 |      Create FFTW3 plans for forward/inverse transforms           |
 |      |                                                            |
 |      v                                                            |
 |  [6] Pre-compute static quantities                               |
 |      - V_loc(G) on the full FFT grid for all species             |
 |      - |G|^2 and G_cart for every FFT grid point                 |
 |      - Nonlocal PP atom data (D_ij, projector metadata)          |
 |      |                                                            |
 |      v                                                            |
 |  [7] Initialize density: superposition of atomic rho(r) from UPF |
 |      (falls back to uniform density if no rho_atomic data)       |
 |      |                                                            |
 |      v                                                            |
 |  +---------------------------------------------------------------+
 |  |                   SCF ITERATION LOOP                          |
 |  |                                                               |
 |  |  [a] rho(r) --FFT--> rho(G)                                  |
 |  |       |                                                       |
 |  |       v                                                       |
 |  |  [b] V_H(G) = 8*pi*rho(G)/|G|^2          (Hartree/Poisson)  |
 |  |       (evaluated on full FFT grid, G^2 <= ecutrho)            |
 |  |       |                                                       |
 |  |       v                                                       |
 |  |  [c] V_xc(r) from rho(r)                 (libxc / built-in)  |
 |  |       (GGA: also compute sigma = |nabla rho|^2 and vsigma)   |
 |  |       |                                                       |
 |  |       v                                                       |
 |  |  [d] V_eff(G) = V_H(G) + N_grid * V_loc(G)  (in G-space)    |
 |  |       V_eff(r) = IFFT(V_eff(G)) + V_xc(r)   (add XC in r)   |
 |  |       |                                                       |
 |  |       v                                                       |
 |  |  [e] For each k-point:                                       |
 |  |       |   Mask psi to per-k active PW set (|k+G|^2 <= ecut)  |
 |  |       |   H|psi> = T|psi> + V_eff*psi + V_NL|psi>            |
 |  |       |   Davidson eigensolver --> {epsilon_nk, psi_nk}       |
 |  |       |                                                       |
 |  |       v                                                       |
 |  |  [f] Fermi level by bisection --> occupations f_nk            |
 |  |       |                                                       |
 |  |       v                                                       |
 |  |  [g] New density: rho(r) = sum_nk w_k*f_nk*|psi_nk(r)|^2    |
 |  |       (normalize to conserve total electron count)            |
 |  |       |                                                       |
 |  |       v                                                       |
 |  |  [h] Total energy = E_band - E_H + E_xc - int(V_xc*n)       |
 |  |       + E_smearing (-TS entropy for metals)                   |
 |  |       |                                                       |
 |  |       v                                                       |
 |  |  [i] Convergence check:                                      |
 |  |       |dE| < energy_threshold AND |dn|_G < density_threshold  |
 |  |       (density norm in G-space to avoid aliasing artifacts)   |
 |  |       |                                                       |
 |  |       v                                                       |
 |  |  [j] Pulay/DIIS density mixing (Kerker for metals)           |
 |  |       Clamp negative density, renormalize charge              |
 |  |       |                                                       |
 |  |       +-- NOT CONVERGED ---> back to [a]                      |
 |  |       +-- CONVERGED -------> exit loop                        |
 |  +---------------------------------------------------------------+
 |      |                                                            |
 |      v                                                            |
 |  [8] Post-SCF                                                    |
 |      - Ewald ion-ion energy (E_real + E_recip + E_self)          |
 |      - Add Ewald energy to total energy                          |
 |      - Hellmann-Feynman forces:                                  |
 |        F = F_ewald + F_local + F_nonlocal                        |
 |      - Symmetrize forces via spglib (if available)               |
 |      |                                                            |
 |      v                                                            |
 |  [9] Output: JSON summary (atomic write via temp + rename)      |
 |                                                                   |
 +===================================================================+
```

---

## 3. Component Diagram

Module dependencies flow top-to-bottom. Each box is a directory under `src/`.

```
                        +----------+
                        |   io/    |
                        | YAML in  |
                        | UPF load |
                        | JSON out |
                        +----+-----+
                             |
                    +--------+--------+
                    |                 |
               +----v----+     +-----v------+
               |  core/  |     |   basis/   |
               | Crystal |     | PlaneWave  |
               | Types   |     | FFTGrid    |
               | Consts  |     | KPoints    |
               +---------+     +-----+------+
                    |                 |
                    +--------+--------+
                             |
                    +--------v--------+
                    |   potential/    |
                    | Hartree  XC    |
                    | LocalPP        |
                    | NonlocalPP     |
                    | Ewald  Forces  |
                    | GGA Gradient   |
                    +--------+-------+
                             |
                    +--------v--------+
                    |  hamiltonian/   |
                    | H|psi> apply   |
                    | (T + Vloc +VNL)|
                    +--------+-------+
                             |
                    +--------v--------+
                    |    solver/      |
                    | Davidson  DIIS |
                    | Fermi    SCF   |
                    | BFGS           |
                    +--------+-------+
                             |
                    +--------v--------+
                    | postprocessing/ |
                    | BandStructure  |
                    | DOS            |
                    +--------+-------+
                             |
                    +--------v--------+
                    |     gpu/        |
                    | FFT  BLAS  Mem |
                    | (stubs or real)|
                    +-----------------+

            +----------+     +----------+
            |  utils/  |     |  main.cpp|
            | Timer    |     | Entry pt |
            | Logger   |     +----------+
            +----------+
```

Dependency rules:
- `core/` depends on nothing (leaf module).
- `basis/` depends on `core/`.
- `io/` depends on `core/` (Crystal, types).
- `potential/` depends on `core/`, `basis/`, `io/` (for `PseudoPotential` data).
- `hamiltonian/` depends on `basis/`, `potential/` (specifically `NonlocalPP`).
- `solver/` depends on everything above; it orchestrates the full calculation.
- `gpu/` is called by `hamiltonian/` and `basis/` but physics code never
  calls vendor APIs directly -- only the `gpu::` namespace.
- `utils/` is available to all modules (timer, logger).

---

## 4. Data Flow Through the SCF Loop

The key data objects and how they flow between modules during each SCF step:

```
  +------------------+
  | Crystal          |   lattice vectors (Mat3), atom positions (Vec3),
  | (core/)          |   volume, reciprocal lattice, frac<->cart transforms
  +--------+---------+
           |
           v
  +------------------+       +------------------+
  | PlaneWaveBasis   |       | PseudoPotential  |
  | (basis/)         |       | (io/upf_parser)  |
  |                  |       |                  |
  | G-vectors: Vec3  |       | beta(r), V_loc(r)|
  | |G|^2 values     |       | D_ij, rho_atom   |
  | kinetic: |k+G|^2 |       | radial mesh      |
  +--------+---------+       +--------+---------+
           |                          |
           v                          v
  +------------------+       +------------------+
  | FFTGrid          |       | LocalPPEvaluator |
  | (basis/)         |       | (potential/)     |
  |                  |       |                  |
  | scatter(PW->grid)|       | V_loc(G): CVec   |
  | gather(grid->PW) |       | E_loc from n(G)  |
  | forward FFT      |       +--------+---------+
  | inverse FFT      |                |
  +--------+---------+       +--------v---------+
           |                 | NonlocalPP       |
           |                 | (potential/)      |
           |                 |                   |
           |                 | beta_i(k+G): CVec |
           |                 | D_ij matrix       |
           |                 | V_NL|psi> via GEMM|
           |                 +--------+----------+
           |                          |
           +------------+-------------+
                        |
                        v
              +---------+----------+
              |    Hamiltonian     |
              |  (hamiltonian/)    |
              |                    |
              |  Input:  psi(G)    |-----> T|psi>: pointwise |k+G|^2 * psi
              |  Output: H|psi(G)> |-----> V_loc|psi>: IFFT, multiply V_eff, FFT
              |                    |-----> V_NL|psi>:  <beta|psi> dot, D*proj
              +--------+-----------+
                       |
                       v
              +--------+-----------+
              |  DavidsonSolver    |
              |  (solver/)         |
              |                    |
              |  Input:  H|.>      |
              |  Output: {eps, psi}|
              +--------+-----------+
                       |
                       v
              +--------+-----------+       +-------------------+
              |   FermiSolver      |       |   Density Output  |
              |   (solver/)        |       |                   |
              |                    |       |  rho(r) = sum     |
              |  Input:  eps_nk    |------>|    w_k*f_nk*      |
              |  Output: f_nk, E_F |       |    |psi_nk(r)|^2  |
              +--------------------+       +--------+----------+
                                                    |
                                                    v
                                           +--------+----------+
                                           |   PulayMixer      |
                                           |   (solver/)       |
                                           |                   |
                                           |  rho_in, rho_out  |
                                           |  --> rho_mixed     |
                                           +-------------------+
```

### Summary of Key Data Objects

| Object | Type | Lives in | Flows to |
|--------|------|----------|----------|
| `psi_nk(G)` | `CVec` (complex128) | Davidson output | density construction, forces |
| `rho(r)` | `RVec` (float64) | real-space grid | XC evaluator, density mixing |
| `rho(G)` | `CVec` | full FFT grid | Hartree solver, local PP energy |
| `V_eff(r)` | `vector<complex_t>` | real-space grid | `Hamiltonian::update_veff` |
| `V_H(G)` | `vector<complex_t>` | full FFT grid | added to V_eff in G-space |
| `V_loc(G)` | `vector<complex_t>` | full FFT grid | added to V_eff in G-space (pre-computed once) |
| `V_xc(r)` | `RVec` | real-space grid | added to V_eff in real space |
| `epsilon_nk` | `double` | per band per k | Fermi solver, band energy |
| `f_nk` | `double` | per band per k | density construction, energy |

### Two Grids: PW Basis vs Full FFT Grid

A critical implementation detail is the distinction between two reciprocal-space
representations:

1. **PW basis** (`PlaneWaveBasis::gvectors()`): G-vectors where `|G|^2 <= ecutwfc`.
   Used for wavefunctions and the eigensolver.

2. **Full FFT grid** (`FFTGrid`, dimensioned for `ecutrho >= 4*ecutwfc`): All grid
   points up to the density cutoff. Used for the Hartree potential, V_loc, and
   density to avoid aliasing from `|psi|^2` products.

The `scatter_to_grid` and `gather_from_grid` methods on `FFTGrid` convert between
these two representations by mapping PW Miller indices `(h,k,l)` to FFT grid
linear indices.

---

## 5. Key Algorithms

### 5.1 Davidson Iterative Eigensolver

Finds the lowest `num_bands` eigenvalues of H without forming the full
Hamiltonian matrix. Implemented in `src/solver/davidson.cpp`.

**Parameters:**
- Max subspace size: `3 * num_bands` (configurable via `subspace_factor`)
- Convergence tolerance: `1e-6` on residual norm
- Max iterations: 100
- Deterministic random seed (42) for reproducibility

**Algorithm:**

```
  Input:  H|.> operator, kinetic diagonal preconditioner, num_bands, num_pw
  Output: {epsilon_n, psi_n} for n = 1..num_bands

  1. Initialize V with num_bands random vectors (seed = 42)
  2. Orthogonalize via modified Gram-Schmidt with reorthogonalization
     (two passes for numerical stability)
  3. Apply H to each basis vector: HV[i] = H|V[i]>
  4. LOOP (max 100 iterations):
     a. Build projected Hamiltonian:  H_ij = <V_i|HV_j>
        (Hermitian: only compute upper triangle)
     b. Diagonalize via LAPACK zheev (complex Hermitian eigensolver)
     c. Compute Ritz vectors: psi_n = sum_i c_ni * V_i
        and H*psi_n = sum_i c_ni * HV_i
     d. Compute residuals:  r_n = H|psi_n> - epsilon_n * |psi_n>
     e. If max(||r_n||) < tolerance: CONVERGED, return
     f. If subspace full (m + num_bands > 3*num_bands):
        RESTART with current Ritz vectors as new basis
     g. Apply kinetic preconditioner:
        t_n[G] = r_n[G] / (|k+G|^2 - epsilon_n)
        with floor |denom| >= 1e-4 to prevent blowup
     h. Orthogonalize t_n against all V (two passes)
     i. If ||t_n|| > 1e-10: normalize, add to V, compute H|t_n>
     j. If no vectors added: stop (preconditioner too exact)
```

### 5.2 Pulay/DIIS Density Mixing

Accelerates SCF convergence by finding the optimal linear combination of
previous density/residual pairs. Implemented in `src/solver/mixing.cpp`.

```
  History depth M = 8, mixing parameter alpha = 0.3

  1. First step: simple linear mixing
     rho_new = rho_in + alpha * (rho_out - rho_in)

  2. Subsequent steps (DIIS):
     a. Store (rho_in, R = rho_out - rho_in) in history deque
     b. Build overlap matrix:  B_ij = <R^(i) | R^(j)>  (real dot product)
     c. Solve augmented constrained system via Gaussian elimination
        with partial pivoting:

        | B_11  B_12  ...  B_1M  1 | | c_1 |   | 0 |
        | B_21  B_22  ...  B_2M  1 | | c_2 |   | 0 |
        | ...                     1 | | ... | = | 0 |
        | B_M1  B_M2  ...  B_MM  1 | | c_M |   | 0 |
        |  1     1    ...   1    0 | | lam |   | 1 |

     d. Optimal density:
        rho_new = sum_i c_i * [rho_in^(i) + alpha * R^(i)]
     e. If matrix is singular (pivot < 1e-15): fall back to using
        only the most recent entry (avoids destroying convergence)
```

**Kerker Preconditioner** (for metals, activated when smearing != None):

Applied to the residual in G-space before Pulay mixing:

```
  R_precond(G) = R(G) * |G|^2 / (|G|^2 + q0^2)     q0 = 1.5 bohr^{-1}
```

This suppresses long-wavelength (small |G|) charge oscillations that cause
"charge sloshing" in metallic systems. At G=0, the filter gives 0, eliminating
uniform charge shifts entirely.

### 5.3 Fermi Level Bisection

Finds the Fermi energy such that the total electron count matches the target.
Implemented in `src/solver/fermi.cpp`.

```
  Input:  eigenvalues[k][n], k-weights, target_electrons, smearing type/width
  Output: Fermi energy E_F, occupation matrix f[k][n]

  1. Set energy bounds: e_min = min(all eigenvalues) - 10*degauss
                        e_max = max(all eigenvalues) + 10*degauss

  2. Bisection loop (max 200 steps, tolerance 1e-10 Ry):
     a. e_mid = (e_min + e_max) / 2
     b. Count electrons:
        N(E_F) = sum_k sum_n spin_factor * w_k * f((epsilon_nk - E_F) / degauss)
     c. If |N - N_target| < 1e-8: converged, exit
     d. If N < N_target: e_min = e_mid
        If N > N_target: e_max = e_mid

  3. Compute final occupations:
     f_nk = spin_factor * f((epsilon_nk - E_F) / degauss)
```

**Supported smearing functions:**

| Type | Formula f(x) | Use case |
|------|-------------|----------|
| None | Step function: 1 if x <= 0, else 0 | Insulators |
| Gaussian | 0.5 * erfc(x) | General metals |
| Marzari-Vanderbilt | Cold smearing (PRL 82, 3296, 1999) | Metals (improved) |
| Fermi-Dirac | 1 / (1 + exp(x)) | Finite-temperature metals |

### 5.4 H|psi> Application (Hamiltonian)

The Kohn-Sham Hamiltonian operator applies three terms to a wavefunction.
Implemented in `src/hamiltonian/hamiltonian.cpp`.

```
  H|psi> = T|psi> + V_eff|psi> + V_NL|psi>

  1. Kinetic (pointwise in G-space):
     (T|psi>)_G = |k+G|^2 * psi_G           [Rydberg units, NOT /2]

  2. Local effective potential (via FFT):
     a. scatter psi_G onto full FFT grid
     b. psi(r) = IFFT(psi_G_grid)
     c. (V*psi)(r) = V_eff(r) * psi(r)       [pointwise real-space multiply]
     d. (V*psi)_G_grid = FFT((V*psi)(r))
     e. gather (V*psi)_G from grid back to PW basis

  3. Nonlocal PP (Kleinman-Bylander):
     a. Precompute beta_i(k+G) for this k-point (cached)
     b. proj_j = <beta_j|psi> = sum_G conj(beta_j(G)) * psi(G)
     c. (V_NL|psi>)_G = sum_{i,j} D_ij * proj_j * beta_i(G)
```

**Per-k masking:** The shared PW basis is expanded to cover all k-points, but
each k-point only uses G-vectors where `|k+G|^2 <= ecutwfc`. The `get_apply_function`
method creates a closure that masks inactive components to zero on input and
applies a high energy wall (`1e4 * psi_G`) on output, pushing the Davidson
solver to converge inactive components to zero amplitude.

### 5.5 Ewald Summation

Computes the electrostatic energy of periodic point charges by splitting
the Coulomb sum into rapidly convergent parts. Implemented in
`src/potential/ewald.cpp`.

```
  E_ion = E_real + E_recip + E_self + E_charged

  E_real  = (1/2) sum'_{T} sum_{i,j} Z_i Z_j erfc(eta*|r_ij+T|) / |r_ij+T|

  E_recip = (4*pi / Omega) sum_{G!=0} |S(G)|^2 * exp(-|G|^2/(4*eta^2)) / |G|^2
            where S(G) = sum_i Z_i * exp(-i G.tau_i)

  E_self  = -(eta / sqrt(pi)) * sum_i Z_i^2

  eta     = sqrt(pi) * (N_atoms / V^2)^(1/6)
```

Forces are the analytical derivatives of each term with respect to atomic
positions, yielding separate real-space and reciprocal-space contributions.

### 5.6 Kleinman-Bylander Nonlocal Pseudopotential

Implemented in `src/potential/nonlocal_pp.cpp`. Each UPF projector with
angular momentum l is expanded into (2l+1) channels for m = -l, ..., +l:

```
  V_NL|psi> = sum_{a,i,j} D_ij^a |beta_i^a> <beta_j^a|psi>

  beta_i(k+G) = (4*pi / sqrt(Omega)) * i^l
                * integral r^2 beta_i(r) j_l(|k+G|*r) dr
                * Y_lm(k+G_hat) * exp(-i(k+G).tau_a)
```

Projectors are precomputed and cached per k-point via `prepare_kpoint()` to
avoid redundant radial Bessel transforms during the Davidson iteration.

---

## 6. Unit Conventions (Rydberg Atomic Units)

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

---

## 7. Source Layout

```
src/
|
+-- core/
|   +-- types.hpp              Fundamental types: Vec3, Mat3, CVec, RVec,
|   |                          CalculationParams, ConvergenceParams, Atom,
|   |                          KPointGrid, enums (CalculationType, SmearingType,
|   |                          EigensolverType)
|   +-- constants.hpp          Physical constants (CODATA 2018): rydberg_to_ev,
|   |                          bohr_to_angstrom, pi, kboltzmann, etc.
|   +-- crystal.hpp/cpp        Crystal class: lattice vectors, reciprocal lattice,
|   |                          cell volume, atom list, frac_to_cart / cart_to_frac
|   +-- element_data.hpp       Periodic table: atomic number, mass, symbol lookup
|   +-- spherical_harmonics.hpp/cpp  Real spherical harmonics Y_lm for KB projectors
|
+-- basis/
|   +-- plane_wave.hpp/cpp     PlaneWaveBasis: enumerates G-vectors satisfying
|   |                          |G|^2 <= ecutwfc (expanded by k_max for multi-k),
|   |                          stores Cartesian G-vectors and |G|^2 norms,
|   |                          computes kinetic energies |k+G|^2
|   +-- fft_grid.hpp/cpp       FFTGrid: FFTW3 wrapper sized for ecutrho, provides
|   |                          forward/inverse FFT, scatter_to_grid (PW -> full grid),
|   |                          gather_from_grid (full grid -> PW), FFT-friendly
|   |                          grid sizing (products of 2, 3, 5)
|   +-- kpoints.hpp/cpp        KPointGenerator: Monkhorst-Pack grid generation with
|                              time-reversal symmetry folding, returns KPointData
|                              (k-points in fractional coords + weights summing to 1)
|
+-- io/
|   +-- input_parser.hpp/cpp   YAML input parser with strict schema validation;
|   |                          unknown keys trigger hard abort; returns ParsedInput
|   |                          containing Crystal + InputData
|   +-- upf_parser.hpp/cpp     UPF v2 pseudopotential reader: parses PP_HEADER,
|   |                          PP_MESH, PP_LOCAL, PP_NONLOCAL (PP_BETA, PP_DIJ),
|   |                          PP_RHOATOM; validates norm conservation;
|   |                          note: PP_BETA stores r*beta(r) and PP_RHOATOM
|   |                          stores 4*pi*r^2*rho(r)
|   +-- output_writer.hpp/cpp  JSON summary writer (atomic write via temp file +
|                              rename to prevent partial output on crash)
|
+-- potential/
|   +-- hartree.hpp/cpp        Poisson solver: V_H(G) = 8*pi*rho(G)/|G|^2
|   |                          (Rydberg units), E_H = (Omega/2) sum conj(V_H)*n
|   +-- xc.hpp/cpp             XC evaluator: libxc wrapper if available, built-in
|   |                          LDA Perdew-Zunger fallback; supports LDA_PZ, LDA_PW,
|   |                          PBE, PBEsol; GGA via vsigma + gradient correction
|   +-- local_pp.hpp/cpp       Local pseudopotential V_loc(G) with Coulomb tail
|   |                          subtraction for numerical stability: separates
|   |                          V_loc(r) into short-range (numerical integral) and
|   |                          long-range analytic (-2Z*erf(r/r_loc)/r) parts
|   +-- nonlocal_pp.hpp/cpp    Kleinman-Bylander nonlocal PP: per-atom projectors
|   |                          expanded in (l,m) channels, Bessel transforms,
|   |                          spherical harmonics, cached per k-point
|   +-- ewald.hpp/cpp          Ewald ion-ion energy: E_real + E_recip + E_self +
|   |                          E_charged, with optimal eta; also computes Ewald forces
|   +-- forces.hpp/cpp         Hellmann-Feynman force calculator: Ewald + local PP
|   |                          + nonlocal PP contributions; F_total = sum of three
|   +-- gradient.hpp/cpp       GGA gradient computation: |nabla rho|^2 (sigma) and
|                              GGA potential correction -2*div(vsigma * nabla n)
|
+-- solver/
|   +-- scf.hpp/cpp            SCF loop orchestrator: builds all components, runs
|   |                          the iteration loop, computes energies via double-
|   |                          counting correction, returns SCFResult with energies,
|   |                          eigenvalues, forces, timing; includes force symmetrization
|   |                          via spglib when available
|   +-- davidson.hpp/cpp       Davidson iterative eigensolver: finds lowest N
|   |                          eigenvalues via subspace expansion (up to 3*N_bands),
|   |                          kinetic diagonal preconditioner, modified Gram-Schmidt
|   |                          with reorthogonalization, LAPACK zheev for subspace
|   +-- mixing.hpp/cpp         Density mixing: LinearMixer (simple alpha blending),
|   |                          PulayMixer (DIIS with 8-step history and Gaussian
|   |                          elimination), KerkerPreconditioner (|G|^2/(|G|^2+q0^2))
|   +-- fermi.hpp/cpp          Fermi level finder by bisection (200 steps, 1e-10 Ry
|   |                          tolerance): supports Gaussian, Marzari-Vanderbilt,
|   |                          Fermi-Dirac, and step-function smearing
|   +-- bfgs.hpp/cpp           BFGS geometry optimizer for ionic relaxation
|
+-- hamiltonian/
|   +-- hamiltonian.hpp/cpp    Kohn-Sham Hamiltonian operator H|psi>:
|                              (1) T|psi>: pointwise |k+G|^2 * psi_G
|                              (2) V_loc|psi>: scatter, IFFT, multiply V_eff, FFT, gather
|                              (3) V_NL|psi>: via NonlocalPP::apply with cached projectors
|                              Provides get_apply_function() with per-k masking and
|                              kinetic_diagonal() for the Davidson preconditioner
|
+-- postprocessing/
|   +-- band_structure.hpp/cpp Non-self-consistent band structure along k-path
|   +-- dos.hpp/cpp            Density of states from eigenvalues + smearing
|
+-- gpu/
|   +-- fft.hpp                GPUFFTGrid: dispatches to cuFFT (CUDA) or rocFFT (HIP)
|   +-- blas.hpp               GPU BLAS: gemm, zdotc via cuBLAS or rocBLAS
|   +-- memory.hpp             GPU memory: gpu_malloc, gpu_free, gpu_memcpy_*
|   +-- gpu_stubs.cpp          CPU-only build stubs: throw GPUNotAvailableError
|
+-- utils/
|   +-- timer.hpp/cpp          KRONOS_TIMER macro, RAII ScopedTimer, TimerRegistry
|   |                          singleton with mutex-protected accumulation and
|   |                          as_map() for JSON output
|   +-- logger.hpp/cpp         Structured JSON logger on stderr: ISO 8601 timestamp,
|   |                          event name, message, arbitrary key-value fields,
|   |                          MPI rank awareness
|   +-- radial_integral.hpp    Simpson rule for radial integrals on UPF meshes
|
+-- main.cpp                   Entry point: CLI argument parsing, banner printing,
                               dispatch to SCF / Relax / Bands / DOS workflows,
                               error handling with distinct exit codes (0-4)
```

---

## 8. GPU Portability

The `kronos::gpu` namespace provides a hardware abstraction layer so that
physics code in `src/hamiltonian/`, `src/basis/`, and `src/solver/` never
calls vendor APIs (CUDA, HIP) directly.

```
  Physics code              GPU abstraction             Vendor backend
  ============              ===============             ==============

  hamiltonian.cpp           gpu::GPUFFTGrid             cuFFT / rocFFT
  (FFT of psi)         -->  .forward()              -->  cufftExecZ2Z()
                             .inverse()                  hipfftExecZ2Z()

  nonlocal_pp.cpp           gpu::gemm()                 cuBLAS / rocBLAS
  (<beta|psi> GEMM)    -->  gpu::zdotc()            -->  cublasZgemm()

  scf.cpp                   gpu::gpu_malloc()           cudaMalloc()
  (memory management)  -->  gpu::gpu_free()         -->  cudaFree()
                             gpu::gpu_memcpy_h2d()       cudaMemcpy()
```

In CPU-only builds (`KRONOS_GPU_BACKEND=none`), `src/gpu/gpu_stubs.cpp`
provides stub implementations that throw `GPUNotAvailableError`. Physics code
uses FFTW3 and CPU BLAS/LAPACK directly, bypassing the GPU layer entirely.

For deterministic GPU results, set `CUBLAS_WORKSPACE_CONFIG=:4096:8`.

### Metal backend (Apple Silicon, v0.5.1)

The Metal backend mirrors the CUDA/HIP pattern, but is fundamentally a
**research/dev tier only**:

- `src/gpu/gpu_context_metal.cpp` — MTLDevice + MTLCommandQueue
- `src/gpu/memory_metal.cpp` — MTLBuffer with `storageModeShared` (Apple
  Silicon's unified memory is exposed as gpu_malloc-compatible pointers
  via a host→MTLBuffer registry)
- `src/gpu/blas_metal.cpp` — complex GEMM dispatched to
  `src/gpu/kernels/complex_gemm.metal`, an MSL `zgemm_fp32` kernel.
  Narrows complex128 → complex64 at the device boundary when
  `apple_fast_mode == true`; otherwise throws so GPUHamiltonian falls
  back to CPU.
- `src/gpu/fft_metal.cpp` — VkFFT (v1.3.4) Metal backend for 3D complex
  FFT in fp32. Same narrow/widen boundary as BLAS.

#### Why fp32 only?

Apple's Metal Shading Language refuses `double` outright across all Metal
versions and toolchain releases — Apple GPUs have no hardware fp64 ALUs.
There is no emulation path. CUDA and HIP retain real fp64 hardware support
and remain the only validation-grade GPU backends.

#### When the Apple GPU path runs

- `hardware.apple_fast_mode: true` in YAML, OR
- `--apple-fast-mode` CLI flag

Both produce a logger warning (`apple_fast_mode` event) at startup. The
validation test suite (`test_validation`) refuses to run when this flag
is on.

See `docs/superpowers/specs/2026-05-16-apple-silicon-metal-backend-design.md`
for the full design rationale.

---

## 9. Observability and Error Handling

### Structured Logging

The `Logger` singleton emits JSON-formatted lines to stderr:

```json
{"timestamp":"2026-03-06T12:34:56Z","level":"info","event":"scf","message":"SCF solver initialized","num_bands":"8","num_pw":"283"}
```

### Scoped Profiling

`KRONOS_TIMER("name")` creates an RAII timer that records elapsed wall time
in the global `TimerRegistry`. Timing data is included in the JSON output
and printed as a summary table at program exit.

### SCF Step Output

Each SCF iteration prints to stdout:

```
SCF step  1: E =  -27.123456 Ry  |dE| = ---        |dn| = 1.23e-02  t = 0.5s
SCF step  2: E =  -27.234567 Ry  |dE| = 1.11e-01  |dn| = 3.45e-03  t = 0.4s
```

### Error Handling

| Error condition | Response |
|----------------|----------|
| SCF non-convergence | Write partial output with `converged: false`, exit code 1 |
| Energy oscillation > 1 Ry (after step 15) | Abort SCF loop with diagnostic |
| UPF parse failure | Hard abort with `UPFParseError` (exit code 3) |
| Input validation failure | Hard abort with `InputValidationError` (exit code 2) |
| LAPACK zheev failure | Throw `runtime_error` from Davidson subspace diag |
| Negative density | Clamp to 0, renormalize to conserve total charge |
| DIIS singular matrix | Fall back to most recent density only |
