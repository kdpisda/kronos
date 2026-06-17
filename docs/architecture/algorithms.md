---
title: Key Algorithms
description: Pseudocode and parameter details for KRONOS's core algorithms — Davidson eigensolver, Pulay/DIIS density mixing, Kerker preconditioner, Fermi bisection, and Ewald summation.
keywords:
  - Davidson eigensolver
  - Pulay mixing
  - DIIS density mixing
  - Kerker preconditioner
  - Fermi bisection
  - plane-wave DFT algorithms
  - Ewald summation
  - Kleinman-Bylander nonlocal pseudopotential
  - Kohn-Sham Hamiltonian
slug: /architecture/algorithms
sidebar_position: 5
---

# Key Algorithms

This page collects pseudocode, parameter tables, and implementation notes for the six core algorithms in KRONOS: the Davidson iterative eigensolver, Pulay/DIIS density mixing with Kerker preconditioning, Fermi level bisection, the Hamiltonian application (`H|ψ>`), Ewald summation, and the Kleinman-Bylander nonlocal pseudopotential. These algorithms live in `src/solver/`, `src/hamiltonian/`, and `src/potential/`; see [Source Layout](source-layout.md) for exact file paths. The [Data Flow](data-flow.md) page shows how the inputs and outputs of these algorithms connect.

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
