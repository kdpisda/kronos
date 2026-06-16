---
title: Source Layout
description: Directory and file map for the KRONOS C++ codebase — what lives in core/, basis/, io/, potential/, solver/, hamiltonian/, postprocessing/, gpu/, and utils/.
keywords:
  - KRONOS source code
  - DFT codebase layout
  - C++ DFT directory structure
  - plane-wave DFT source
  - KRONOS files
  - C++20 DFT engine
slug: /architecture/source-layout
sidebar_position: 7
---

# Source Layout

The KRONOS source tree is organized by physical/functional module rather than by type, so that related headers and implementations sit together. The dependency rules described in the [Component Diagram](component-diagram.md) are enforced structurally: `core/` has no upstream includes, `gpu/` is a leaf called only from `hamiltonian/` and `basis/`, and `utils/` is available everywhere. This page lists every file with a one-line description of its responsibility.

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
