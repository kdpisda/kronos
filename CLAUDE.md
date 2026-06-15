# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**KRONOS** (Kohn-Residual Optimized Numerics Over Silicon) is a research-grade, ab initio plane-wave Density Functional Theory (DFT) engine. It computes ground-state total energy, electronic density, Kohn-Sham eigenvalues, and ionic forces for periodic crystalline systems.

- **Language:** C++20 core, CUDA/HIP for GPU kernels, Python (pybind11) for scripting interface
- **License:** GPL v3
- **Build system:** CMake 3.20+
- **Target accuracy:** < 2 meV/atom Delta test score vs Wien2k all-electron reference

## Architecture

### Core Computation: SCF Loop

The self-consistent field loop is the central algorithm:
1. Parse YAML input + load UPF pseudopotential files
2. Build `PlaneWaveBasis`: enumerate G-vectors where |k+G|²/2 ≤ ecutwfc; build FFT grids
3. Detect k-point symmetry via spglib; generate irreducible Brillouin zone k-points
4. Initialize electron density from superposition of atomic densities
5. SCF iterations:
   - Hartree potential: Poisson equation in G-space V_H(G) = 4π n(G) / |G|²
   - XC potential: V_xc(r) via libxc on real-space grid
   - Apply Hamiltonian H|ψ⟩ per k-point via FFT (the GPU hot path)
   - Eigensolver: Davidson (primary) or LOBPCG (fallback)
   - Fermi level by bisection; compute occupations
   - New density from wavefunctions; check convergence
   - Pulay/DIIS density mixing
6. Post-convergence: forces, stress tensor, total energy
7. Output: JSON summary + HDF5 files

### GPU Hot Path: H|ψ⟩ Application

```
Kinetic:   T|ψ⟩_G = |k+G|²/2 · ψ_G          [pointwise multiply]
Local:     ψ_r = IFFT(ψ_G)                    [cuFFT/rocFFT]
           V·ψ_r = V_eff(r) · ψ_r             [pointwise multiply]
           V·ψ_G = FFT(V·ψ_r)                 [cuFFT/rocFFT]
Non-local: proj_i = <β_i|ψ> via GEMM          [cuBLAS/rocBLAS]
           V_NL|ψ⟩ via GEMM                   [cuBLAS/rocBLAS]
```

### Source Layout

- `src/core/` — Types (`types.hpp`), physical constants, `Crystal` class, element data
- `src/basis/` — `PlaneWaveBasis` (G-vector enumeration), `FFTGrid` (FFTW3 wrapper)
- `src/io/` — YAML input parser, UPF pseudopotential parser, JSON output writer
- `src/potential/` — Hartree, XC (libxc wrapper + built-in LDA fallback), local/nonlocal PP, Ewald, GGA gradients, forces
- `src/solver/` — Davidson eigensolver, Pulay/DIIS mixing, Fermi level bisection, SCF loop, BFGS geometry optimizer
- `src/postprocessing/` — Band structure calculator, DOS calculator
- `src/hamiltonian/` — H|ψ⟩ application (kinetic + local via FFT + nonlocal via GEMM)
- `src/gpu/` — GPU abstraction layer (`fft.hpp`, `blas.hpp`, `memory.hpp`); stubs in CPU-only builds
- `src/utils/` — Scoped timer/profiling registry, structured JSON logger
- `test/` — GoogleTest suite: test_input, test_basis, test_fft, test_upf, test_solvers, test_physics

### Key External Libraries

| Library | Purpose |
|---------|---------|
| FFTW3 (CPU) / cuFFT / rocFFT (GPU) | Fast Fourier transforms |
| BLAS+LAPACK (CPU) / cuBLAS / rocBLAS (GPU) | Dense linear algebra |
| ELPA | Distributed parallel eigenvalue solver (v2022.11+ for GPU) |
| libxc | Exchange-correlation functionals (v6.0+ required) |
| spglib | Space group symmetry detection, k-point generation |
| HDF5 | Binary output for density, wavefunctions, restart |
| yaml-cpp | YAML input file parsing |
| pybind11 | Python bindings |
| MPI (OpenMPI/MPICH) | Distributed memory parallelism |

## Build Commands

```bash
# CPU-only build (v0.1 default)
cmake -B build -S .
cmake --build build -j$(nproc)

# Run
./build/kronos examples/si_bulk.yaml

# Run tests
cd build && ctest --output-on-failure

# Run a single test
./build/test_basis --gtest_filter='PlaneWaveBasis.SiBulkBasisSize'

# CUDA build
cmake -B build -S . -DKRONOS_GPU_BACKEND=cuda
cmake --build build -j$(nproc)

# HIP/AMD build
cmake -B build -S . -DKRONOS_GPU_BACKEND=hip -DROCM_PATH=/opt/rocm

# Metal build (Apple Silicon -- macOS 13+, Xcode.app + Metal Toolchain required)
cmake -B build -S . -DKRONOS_GPU_BACKEND=metal
cmake --build build -j$(sysctl -n hw.ncpu)
```

> **Apple Silicon Metal backend — research/dev tier only.** Runs entirely in
> fp32 (Apple's MSL has no `double`). NOT validation-grade. Activate with
> `hardware.apple_fast_mode: true` in YAML or `--apple-fast-mode` CLI flag;
> without it, KRONOS falls back to CPU even on a Metal build. Requires
> Xcode.app, not just CLT.

### CMake Options

| Option | Default | Description |
|--------|---------|-------------|
| `KRONOS_GPU_BACKEND` | `none` | GPU backend: `none`, `cuda`, `hip`, or `metal` |
| `KRONOS_BUILD_TESTS` | `ON` | Build GoogleTest test suite |
| `KRONOS_BUILD_PYTHON` | `OFF` | Build pybind11 Python bindings |

### Required Dependencies

FFTW3, BLAS, LAPACK, yaml-cpp. Optional: HDF5, MPI, libxc (built-in LDA fallback if absent).

## Input/Output Format

- **Input:** YAML file (`kronos.yaml`) with crystal structure, calculation parameters, pseudopotential paths, hardware config
- **Output:** JSON summary + HDF5 files for density/wavefunctions
- Unknown YAML keys must raise errors (strict schema, no silent ignoring)

## Numerical Constraints (Hard Rules)

- **Always float64/complex128** for wavefunction coefficients — never float32
  *(sole exception: the Apple Metal backend when `apple_fast_mode` is explicitly
  enabled by the user; this opt-in narrows at the device boundary and is
  explicitly NOT validation-grade — the validation suite refuses to run in
  this mode)*
- ecutrho must be ≥ 4 × ecutwfc (norm-conserving PP) or 12 × ecutwfc (PAW)
- Negative electron density: clamp to 0 with warning; abort if > 1e-6
- Energy oscillation > 1 Ry between consecutive SCF steps: abort with diagnostic
- Max 200 SCF iterations; ecutwfc range 10–500 Ry
- Davidson subspace size: 3 × N_bands
- DIIS mixing history: 8 steps
- HDF5 output: write atomically via temp file + rename (never partial writes)
- Pseudopotential norm-conservation check is mandatory on load

## GPU Portability

The `gpu::` namespace wraps CUDA/HIP calls so physics code never calls vendor APIs directly. This is the abstraction boundary — adding AMD support means only touching `src/gpu/` files.

For deterministic GPU results: `CUBLAS_WORKSPACE_CONFIG=:4096:8`

## Observability

- SCF step output: energy, |dE|, |dn|, wall time per step
- Structured JSON logs on stderr with fields: timestamp, event, scf_step, wall_s, gpu_mem_mb, mpi_rank
- NVTX ranges on all GPU kernels for Nsight Systems profiling
- MPI timing via PMPI wrapper (Score-P compatible)

## Checkpoint/Restart

Checkpoints written every N SCF steps to HDF5: wavefunctions, density, DIIS history, step counter. On restart, input hash is verified against checkpoint — mismatch warns but allows override.

## Regression Test Systems

1. Si bulk (diamond, LDA) — energy, forces, stress, band gap
2. Cu FCC (metal, smearing, PBE) — Fermi level, DOS
3. H₂O molecule (Gamma-only, GGA) — forces, geometry optimization
4. Fe BCC (spin-polarized LSDA) — magnetic moment ~2.2 μ_B
5. MgO rocksalt (ionic, PAW) — PAW one-center energy
6. Graphene (2D, vacuum padding) — band structure Dirac cone

## Versioning Roadmap

- **v0.1 (current):** LDA/GGA NCPP with k-point sampling, CPU only
- **v0.2:** MPI parallelization
- **v0.5:** GPU offloading (CUDA/HIP)
- **v0.8:** PAW support
- **v1.0:** Full production release with Python package, CI/CD, docs
- **v2.0:** Hybrid functionals (HSE06/PBE0), non-collinear magnetism

## Error Handling Conventions

- SCF non-convergence: write partial output with `converged: false`, suggest fixes
- GPU OOM: auto-fallback to CPU with warning
- UPF parse failure: hard abort with file path, line number, and download URL
- Davidson divergence (residual > 1e3): auto-switch to LOBPCG for that k-point
- Input validation failures: hard abort with field name and allowed values
