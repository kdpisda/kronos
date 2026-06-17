---
title: Numerical Precision in Plane-Wave DFT
description: Why plane-wave DFT requires double precision, FFT round-trip error, eigenvalue convergence, density-norm tolerances, and the meV/atom target in KRONOS.
keywords:
  - DFT numerical precision
  - double precision DFT
  - FFT numerical error
  - eigenvalue convergence DFT
  - density convergence DFT
  - meV/atom accuracy
  - fp64 DFT
  - plane-wave DFT precision
  - SCF tolerances
  - DFT validation
slug: /physics/numerical-precision
sidebar_position: 14
---

# Numerical Precision in Plane-Wave DFT

Total energies in DFT are differences of large numbers. A typical bulk solid has individual energy components (kinetic, Hartree, exchange-correlation, ion-ion) on the order of tens of Rydberg; their sum lands within a few hundred mRy of the true value, and meaningful chemistry happens at the meV/atom level. To resolve a 1 meV/atom difference between two structural candidates, every term in the energy sum must be accurate to better than $10^{-7}$ in relative precision. This is why plane-wave DFT is a double-precision discipline — and why the Apple Silicon Metal backend in KRONOS is research/dev tier only.

## The accuracy target

KRONOS aims for the **Δ-test** standard used by the production DFT community: total energies should agree with reference codes (Quantum ESPRESSO, VASP, Wien2k) to **< 2 meV/atom** on common benchmark systems. On the Si LDA Γ-only benchmark, KRONOS hits **0.07 meV/atom** vs QE — comfortably inside the bar. The 4×4×4 shifted-grid number is 0.15 meV/atom (with density symmetrization).

These numbers are not aspirational; they are the assertion targets in `test/test_validation.cpp::QEValidation.*`. Every CI build runs them.

To stay below 2 meV/atom, the floating-point error budget must be:

| Source | Relative error | Absolute error on E ~ 30 Ry |
|---|---|---|
| FFT round-trip | $\sim 10^{-14}$ | $\sim 10^{-11}$ Ry |
| Eigensolver residual | $\leq 10^{-8}$ | $\sim 10^{-7}$ Ry |
| SCF density tolerance | $\leq 10^{-6}$ | $\sim 10^{-6}$ Ry |
| **Sum** | — | **$\sim 10^{-6}$ Ry per atom** |

That's the room we have. Drop FFT to single precision (fp32 with $\sim 10^{-7}$ relative) and the kinetic energy term alone blows the budget by 5 orders of magnitude.

## Why double precision (fp64)

Floating-point arithmetic in IEEE 754:

- **fp32 ("single")**: ~7 decimal digits of relative precision
- **fp64 ("double")**: ~16 decimal digits

A 64³ FFT has $\sim 10^5$ operations per output element. With fp32 each operation drops a fraction of a digit; after $10^5$ ops we're at $10^{-2}$–$10^{-3}$ relative error — millions of times worse than the Δ-test budget. With fp64 the same chain finishes at $10^{-11}$–$10^{-12}$, comfortable.

KRONOS hard-codes `complex128` (= `std::complex<double>`) for wavefunction coefficients throughout. The single exception is the Apple Silicon Metal backend, where the hardware refuses to compile `double` in MSL — that path is therefore fp32 only and explicitly gated as research/dev tier (it cannot be used for validation runs).

## FFT round-trip error

A correctness-critical test is the FFT round-trip: $\mathrm{IFFT}(\mathrm{FFT}(\psi))$ should recover $\psi$ to machine precision. KRONOS asserts this in unit tests:

| Grid size | fp64 round-trip error (per element) |
|---|---|
| 16³ | $\sim 10^{-14}$ |
| 32³ | $\sim 10^{-13}$ |
| 64³ | $\sim 10^{-12}$ |
| 128³ | $\sim 10^{-11}$ |

The slow growth ($\sim N \log N$ in the worst case) is the expected behavior for radix-2 Cooley-Tukey. For comparison, the fp32 path on the same 64³ grid lands at $\sim 10^{-5}$ — usable for visualization, not for energies.

KRONOS uses FFTW3 on the CPU (best-in-class accuracy, plans cached per grid), cuFFT on NVIDIA (double-double accumulation when requested), rocFFT on AMD, and VkFFT on Apple Silicon (fp32 only there).

## Eigenvalue convergence

The Davidson iteration converges when the residual norm $\|H|\psi\rangle - \varepsilon |\psi\rangle\|$ drops below a tolerance. KRONOS uses $10^{-8}$ Ry by default, scaling with the eigenvalue magnitude. The exact convergence behavior depends on:

- Subspace dimension (KRONOS uses 3× the number of bands)
- Preconditioner quality (the kinetic-energy diagonal preconditioner has a floor of $|\text{denom}| \geq 10^{-4}$ to prevent blowup at $\varepsilon_n \approx |\mathbf{k}+\mathbf{G}|^2$)
- Initial guess quality (random with seed=42 for reproducibility)

When Davidson diverges (residual > $10^3$ Ry) — typically with bad pseudopotentials or pathological inputs — KRONOS auto-switches to LOBPCG for that k-point.

## Density tolerance and energy oscillation

The SCF loop stops when both $|\Delta E| < \epsilon_E$ AND $\|n_\mathrm{out} - n_\mathrm{in}\|_G < \epsilon_n$. Defaults: $\epsilon_E = 10^{-7}\,\mathrm{Ry}$ and $\epsilon_n = 10^{-6}$. The G-space density norm avoids aliasing artifacts that the real-space norm would have at the boundary of the FFT grid.

A common failure mode is **energy oscillation**: $E^{(k)}$ goes up and down by 1–10 mRy between iterations and never converges. KRONOS detects oscillation of more than 1 Ry between consecutive SCF steps and aborts with a diagnostic. The usual fix is a smaller mixing parameter $\alpha$ or a tighter Fermi-level bisection tolerance.

## The Fermi-bisection trap

The Fermi level $\varepsilon_F$ is found by bisection with tolerance $10^{-10}$ Ry. Tighter tolerances ($10^{-13}$, say) sound better but cause SCF oscillation with `smearing: None` on toy pseudopotentials — the Fermi level shifts by floating-point noise between iterations, occupations seesaw, and the density refuses to settle. The $10^{-10}$ value is empirically tuned to be tight enough not to limit total-energy accuracy but loose enough to be stable.

## Pseudopotential norm conservation

UPF files report a "valence charge" $Z_\mathrm{val}$. The integrated atomic density should equal this number; KRONOS validates at parse time that

$$\left| \int_0^\infty 4\pi r^2 \rho_a^\mathrm{atom}(r)\, dr - Z_\mathrm{val} \right| < 10^{-3}$$

A failed check aborts with the file path, line number, and a suggested replacement from the QE pseudopotential library.

## Deterministic GPU execution

For GPU-built KRONOS, set the cuBLAS workspace config to enforce reproducible GEMM order:

```
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

Without this, parallel reductions can deliver the same answer with $\pm 1$ ULP variation between runs — fine for production but breaks bit-for-bit regression testing.

## Reproducibility across runs

KRONOS uses a fixed RNG seed (`seed = 42`) for Davidson initial guesses. Same input → same intermediate trajectory → same final energy to all digits, on the same hardware. Cross-platform agreement (CPU ↔ CUDA ↔ HIP) is typically within $10^{-8}$ Ry; cross-Apple-Silicon-fp32 is much looser (~$10^{-3}$ Ry).

## How KRONOS enforces these

- `complex_t = std::complex<double>` is a global type alias; no fp32 in physics code paths.
- FFT round-trip tests on 16³, 32³, 64³ grids run in `test/test_fft.cpp`.
- Davidson convergence asserted in `test/test_solvers.cpp`.
- SCF tolerances exposed via YAML; defaults match the Δ-test budget.
- QE-comparison regression tests in `test/test_validation.cpp::QEValidation.*` fail CI if any benchmark drifts by more than 0.5 meV/atom.

## References

- IEEE Standard for Floating-Point Arithmetic, IEEE Std 754-2019.
- Higham, N. J. *Accuracy and Stability of Numerical Algorithms*, 2nd ed., SIAM, 2002.
- Lejaeghere, K. *et al.* "Reproducibility in density functional theory calculations of solids", *Science* **351**, aad3000 (2016) — the Δ-test paper.
- Goedecker, S. & Hoisie, A. *Performance Optimization of Numerically Intensive Codes*, SIAM, 2001.
- Frigo, M. & Johnson, S. G. "The design and implementation of FFTW3", *Proc. IEEE* **93**, 216 (2005).
