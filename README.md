# KRONOS

**Kohn-Residual Optimized Numerics Over Silicon** -- a research-grade, ab initio plane-wave Density Functional Theory (DFT) engine.

KRONOS computes ground-state total energy, electronic density, Kohn-Sham eigenvalues, and ionic forces for periodic crystalline systems using norm-conserving pseudopotentials with LDA and GGA exchange-correlation functionals.

## Features

- **Plane-wave basis** with configurable energy cutoff (10-500 Ry)
- **LDA** (Perdew-Zunger) and **GGA** (PBE, PBEsol) exchange-correlation via built-in routines or libxc
- **UPF v2 pseudopotentials** (norm-conserving)
- **Davidson eigensolver** with Pulay/DIIS density mixing
- **Monkhorst-Pack k-point sampling** with time-reversal symmetry reduction
- **Ewald summation** for ion-ion interactions
- **Hellmann-Feynman forces** and **BFGS geometry optimization**
- **Band structure** and **density of states** post-processing
- **JSON** summary output and **HDF5** binary output
- **GPU-ready architecture** (CUDA/HIP abstraction layer, v0.5+)

## Requirements

- C++20 compiler (GCC 11+, Clang 14+, Apple Clang 15+)
- CMake 3.20+
- FFTW3 (double precision)
- BLAS + LAPACK
- yaml-cpp

**Optional:** HDF5, MPI, libxc (v6.0+; built-in LDA fallback if absent)

### macOS (Homebrew)

```bash
brew install cmake fftw lapack yaml-cpp hdf5
```

### Ubuntu/Debian

```bash
sudo apt install cmake libfftw3-dev liblapack-dev libyaml-cpp-dev libhdf5-dev
```

## Building

```bash
# Configure and build (CPU-only, default)
cmake -B build -S .
cmake --build build -j$(nproc)

# Run the test suite
cd build && ctest --output-on-failure
```

### CMake Options

| Option | Default | Description |
|--------|---------|-------------|
| `KRONOS_GPU_BACKEND` | `none` | GPU backend: `none`, `cuda`, or `hip` |
| `KRONOS_BUILD_TESTS` | `ON` | Build GoogleTest test suite |
| `KRONOS_BUILD_PYTHON` | `OFF` | Build pybind11 Python bindings |

### GPU Builds

```bash
# CUDA
cmake -B build -S . -DKRONOS_GPU_BACKEND=cuda
cmake --build build -j$(nproc)

# HIP/AMD
cmake -B build -S . -DKRONOS_GPU_BACKEND=hip -DROCM_PATH=/opt/rocm
cmake --build build -j$(nproc)
```

## Usage

```bash
./build/kronos examples/si_bulk.yaml
```

### Input Format

KRONOS uses YAML input files with strict schema validation (unknown keys are rejected):

```yaml
system:
  lattice: [[5.43, 0, 0], [0, 5.43, 0], [0, 0, 5.43]]
  atoms:
    - {symbol: Si, position: [0.0, 0.0, 0.0]}
    - {symbol: Si, position: [0.25, 0.25, 0.25]}

calculation:
  type: scf          # scf | relax | bands | dos
  ecutwfc: 60.0      # wavefunction cutoff (Ry)
  ecutrho: 240.0     # density cutoff (Ry), default = 4 * ecutwfc
  kpoints: [8, 8, 8, 0, 0, 0]  # Monkhorst-Pack grid + shift
  xc: PBE            # LDA_PZ | LDA_PW | PBE | PBEsol
  smearing: marzari-vanderbilt  # gaussian | fermi-dirac | marzari-vanderbilt
  degauss: 0.01      # smearing width (Ry)

pseudopotentials:
  Si: Si.ONCVPSP.upf

convergence:
  energy: 1e-8       # energy threshold (Ry)
  density: 1e-9      # density threshold
  max_scf_steps: 100
```

### Output

- **JSON summary** with total energy, eigenvalues, forces, and convergence status
- **HDF5 files** for density and wavefunctions (atomic write via temp + rename)
- **Structured JSON logs** on stderr with timing and GPU memory usage

## Project Structure

```
src/
  core/        Types, constants, Crystal class, element data
  basis/       PlaneWaveBasis, FFTGrid, k-point generation
  io/          YAML input parser, UPF parser, JSON output writer
  potential/   Hartree, XC, local/nonlocal PP, Ewald, forces, GGA gradients
  solver/      Davidson eigensolver, Pulay/DIIS mixing, Fermi level, SCF loop, BFGS
  hamiltonian/ H|psi> application (kinetic + local + nonlocal)
  postprocessing/  Band structure, density of states
  gpu/         GPU abstraction layer (CUDA/HIP stubs for CPU builds)
  utils/       Timer/profiling, structured logger
test/          GoogleTest suite (258 tests)
examples/      Example input files
cmake/         CMake find modules (FFTW3, LibXC)
```

## Testing

The test suite covers all major subsystems with 258 tests:

```bash
cd build && ctest --output-on-failure

# Run a single test
./build/test_basis --gtest_filter='PlaneWaveBasis.SiBulkBasisSize'
```

| Test File | Tests | Coverage |
|-----------|-------|----------|
| `test_input` | 29 | YAML parsing, validation, edge cases |
| `test_crystal` | 19 | Lattice, volume, coordinate transforms |
| `test_basis` | 17 | G-vectors, kinetic energies, cutoff |
| `test_fft` | 11 | Forward/inverse, Parseval, scatter/gather |
| `test_upf` | 24 | UPF parsing, validation, round-trip |
| `test_hamiltonian` | 12 | Kinetic, Hermiticity, linearity |
| `test_solvers` | 19 | Davidson, Pulay mixing, Fermi level |
| `test_physics` | 29 | Hartree, XC, Ewald, spherical harmonics |
| `test_scf` | 25 | SCF convergence, energy components |
| `test_postprocessing` | 16 | Band structure, DOS |
| `test_forces` | 10 | Hellmann-Feynman forces, BFGS |
| `test_output` | 10 | JSON output format, atomic writes |
| `test_gradient` | 7 | GGA sigma, potential correction |
| `test_utils` | 14 | Timer, logger, constants |
| `test_unimplemented` | 16 | Skip stubs for future features |

## Roadmap

| Version | Features |
|---------|----------|
| **v0.1** (current) | LDA/GGA NCPP, k-point sampling, CPU only |
| v0.2 | MPI parallelization |
| v0.5 | GPU offloading (CUDA/HIP) |
| v0.8 | PAW support |
| v1.0 | Production release, Python package, CI/CD |
| v2.0 | Hybrid functionals (HSE06/PBE0), non-collinear magnetism |

## License

GPL v3 -- see [LICENSE](LICENSE).
