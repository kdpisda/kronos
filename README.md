# KRONOS

**Kohn-Residual Optimized Numerics Over Silicon** -- a research-grade, ab initio plane-wave Density Functional Theory (DFT) engine.

KRONOS computes ground-state total energy, electronic density, Kohn-Sham eigenvalues, and ionic forces for periodic crystalline systems using norm-conserving pseudopotentials with LDA and GGA exchange-correlation functionals.

## Features

- **Plane-wave basis** with configurable energy cutoff (10-500 Ry)
- **LDA** (Perdew-Zunger) and **GGA** (PBE, PBEsol) exchange-correlation via built-in routines or libxc
- **UPF v2 pseudopotentials** (norm-conserving)
- **Davidson eigensolver** with Pulay/DIIS density mixing
- **Monkhorst-Pack k-point sampling** with spglib space-group symmetry IBZ reduction
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
test/          GoogleTest suite (288 tests)
examples/      Example input files
docs/          Architecture, user guide, developer guide, API reference, physics notes
cmake/         CMake find modules (FFTW3, LibXC)
```

## Documentation

See the `docs/` directory for detailed documentation:

- [Architecture](docs/architecture.md) -- SCF flowchart, component diagram, data flow, source map
- [User Guide](docs/user_guide.md) -- Quick start, YAML reference, examples, troubleshooting
- [Developer Guide](docs/developer_guide.md) -- Directory layout, adding features, test conventions
- [API Reference](docs/api_reference.md) -- All public classes/structs by module
- [Physics Notes](docs/physics_notes.md) -- KS-DFT, PW formalism, pseudopotentials, Ewald

## Testing

The test suite covers all major subsystems with 264+ tests:

```bash
cd build && ctest --output-on-failure

# Run a single test
./build/test_basis --gtest_filter='PlaneWaveBasis.SiBulkBasisSize'
```

| Category | Tests | Description |
|----------|-------|-------------|
| Unit tests | 200+ | Basis, FFT, parsing, potentials, solvers, Hamiltonian |
| Physics invariants | 14 | Hermiticity, sum rules, symmetry, Parseval |
| Convergence studies | 8 | Ecut convergence, k-grid convergence, mixing |
| Regression baselines | 8 | Frozen energy values, Ewald, forces, Madelung |
| Forces | 12 | Real PP FD validation, Ewald FD, Newton's 3rd law, BFGS |
| Multi-system | 4 | Al FCC, Cu FCC (d-electrons), force validation |
| Validation | 18 | QE-matched Si diamond, physics checks |

## Validation

KRONOS has been validated against Quantum ESPRESSO for silicon diamond with the `Si.pz-vbc.UPF` pseudopotential (Perdew-Zunger LDA, norm-conserving):

| Configuration | KRONOS | QE Reference | Difference |
|---------------|--------|-------------|------------|
| Gamma-only, ecut=12 | -14.51875 Ry | -14.51876 Ry | 10 uRy (**0.07 meV/atom**) |
| 4x4x4 shifted, ecut=18 | -15.8439 Ry | -15.8445 Ry | 0.6 mRy (4.2 meV/atom) |
| Ewald energy | -16.8998 Ry | -16.8998 Ry | < 10 uRy |

All energy components (kinetic, Hartree, XC, nonlocal PP) individually agree with QE.
The Gamma-only result matches to single-digit micro-Rydberg precision, confirming the
core algorithms are essentially exact.

Hellmann-Feynman forces validated against finite-difference to **5 significant figures**.
Multi-system validation passes for Al FCC (sp-metal) and Cu FCC (d-metal) in addition
to Si diamond. See [VALIDATION.md](VALIDATION.md) for details.

```bash
# Reproduce the validation calculation
./build/src/kronos examples/si_qe_validation.yaml
```

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
