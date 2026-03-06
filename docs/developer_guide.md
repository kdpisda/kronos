# KRONOS Developer Guide

## 1. Directory Layout and Module Organization

```
kronos/
├── CMakeLists.txt              # Root: project options, find_package, subdirs
├── src/
│   ├── CMakeLists.txt          # Builds kronos_lib (static) + kronos executable
│   ├── main.cpp                # Entry point: parse YAML, run SCF, write JSON
│   ├── core/                   # types.hpp, constants.hpp, crystal, element_data, spherical_harmonics
│   ├── basis/                  # plane_wave (G-vector enumeration), fft_grid (FFTW3), kpoints (MP grid)
│   ├── io/                     # input_parser (YAML), upf_parser (UPF), output_writer (JSON/HDF5)
│   ├── potential/              # hartree, xc, local_pp, nonlocal_pp, ewald, gradient, forces
│   ├── solver/                 # scf, davidson, mixing (Pulay/DIIS), fermi, bfgs
│   ├── hamiltonian/            # H|psi> application: kinetic + local (FFT) + nonlocal (GEMM)
│   ├── postprocessing/         # band_structure, dos
│   ├── gpu/                    # fft.hpp, blas.hpp, memory.hpp, gpu_stubs.cpp
│   └── utils/                  # timer (KRONOS_TIMER macro), logger (structured JSON)
├── test/                       # GoogleTest suite (15 executables)
│   ├── CMakeLists.txt          # FetchContent(googletest v1.14), kronos_add_test()
│   ├── test_helpers.hpp        # Shared crystals, pseudopotentials, tolerances
│   └── test_*.cpp              # One file per module
├── examples/                   # YAML input files
├── pseudopotentials/           # UPF files
└── docs/                       # Documentation
```

Each `src/` subdirectory contains `.hpp`/`.cpp` pairs. Headers use module-relative includes: `#include "core/types.hpp"`. Adding a `.cpp` to any `src/` subdirectory is picked up automatically by `GLOB_RECURSE` in `src/CMakeLists.txt` (re-run cmake after adding files).

The dependency flow is strictly layered: `core` <- `basis` <- `potential` <- `hamiltonian` <- `solver` <- `io/main`. The `gpu/` namespace is the sole bridge to vendor APIs. `utils/` is used everywhere.


## 2. How to Add a New XC Functional

**Step 1.** In `src/potential/xc.cpp`, locate `XCEvaluator::init_functional()` and add a branch:

```cpp
} else if (name_ == "BLYP") {
    is_gga_ = true;
#ifdef KRONOS_HAS_LIBXC
    x_func_id_ = XC_GGA_X_B88;
    c_func_id_ = XC_GGA_C_LYP;
#endif
}
```

**Step 2.** Set `is_gga_ = true` for GGA functionals. The SCF loop checks `is_gga()` to compute density gradients via `GGAGradient` and call `evaluate_gga()` instead of `evaluate()`.

**Step 3.** Register the name as an allowed value in `src/io/input_parser.cpp`.

**Step 4.** (Optional) For libxc-free operation, add analytic implementations alongside `builtin::lda_x` and `builtin::lda_c_pz` in `xc.cpp`, routing to them in the `#else` path.

**Step 5.** Add tests:

```cpp
TEST(XCFunctional, BLYPSmokeTest) {
    kronos::XCEvaluator xc("BLYP");
    EXPECT_TRUE(xc.is_gga());
    RVec rho(100, 0.01), sigma(100, 1e-4);
    auto result = xc.evaluate_gga(rho, sigma, 100.0);
    EXPECT_LT(result.energy, 0.0);
}
```

Currently supported: `LDA_PZ`, `LDA_PW`, `PBE`, `PBEsol`.


## 3. How to Add a New Potential Type

**Step 1.** Create `src/potential/your_potential.hpp` and `.cpp`:

```cpp
#pragma once
#include "core/types.hpp"
#include "core/crystal.hpp"
#include "basis/plane_wave.hpp"
namespace kronos {
class YourPotential {
public:
    YourPotential(const Crystal& crystal, const PlaneWaveBasis& basis);
    [[nodiscard]] CVec compute(const CVec& density_g) const;   // V(G)
    [[nodiscard]] double energy(const CVec& density_g) const;   // Ry
private:
    const Crystal& crystal_;
    const PlaneWaveBasis& basis_;
};
} // namespace kronos
```

**Step 2.** Integrate into `src/solver/scf.cpp`:
- Construct alongside other potentials at the start of `solve()`.
- Call `compute()` in the SCF iteration and add to `V_eff`.
- Call `energy()` and include in the total energy decomposition.
- Add a field to `SCFResult` in `scf.hpp` if separately reported.

**Step 3.** Re-run cmake (auto-globbed). Write tests, register with `kronos_add_test()`.

If the potential contributes to forces, add derivatives in `src/potential/forces.cpp` following the `compute_local_forces()` / `compute_nonlocal_forces()` pattern.


## 4. Test Conventions and How to Add Tests

### Shared Helpers (test/test_helpers.hpp)

`kronos::test` provides factories and tolerance constants:

| Helper | Description |
|--------|-------------|
| `make_si_diamond_crystal()` | Si diamond FCC, a=5.43 A, 2 atoms |
| `make_nacl_crystal()` | NaCl rocksalt, 8 atoms |
| `make_cscl_crystal()` | CsCl primitive, 2 atoms |
| `make_si_pp_map()` | Local-only Gaussian Si pseudopotential |
| `make_si_pp_map_nonlocal()` | Si PP with p-type KB projector, D=-2.0 Ry |
| `make_si_diamond_displaced(delta, atom, dir)` | Displaced crystal for force finite-difference tests |

Tolerances: `ENERGY_TOL` (1e-6), `FORCE_TOL` (1e-4), `FFT_TOL` (1e-10), `TIGHT_TOL` (1e-12), `LOOSE_TOL` (1e-3).

### Test Categories

- **Regression** (`test_regression.cpp`): Frozen baseline energies as `constexpr double`. `SetUpTestSuite()` runs SCF once and shares the `SCFResult`.
- **Physics** (`test_physics.cpp`): Physical invariants -- Hermiticity, force sum rules, sign of energy components.
- **Convergence** (`test_convergence.cpp`): Energy vs cutoff monotonicity, k-grid convergence, SCF iteration limits.
- **Validation** (`test_validation.cpp`): Comparison against Quantum ESPRESSO references in `test/reference/`.

### Adding a New Test

1. Create `test/test_yourmodule.cpp`.
2. Add `kronos_add_test(test_yourmodule)` to `test/CMakeLists.txt`.
3. Build and run: `cmake --build build -j$(nproc) && ./build/test_yourmodule`
4. Use `GTEST_SKIP()` if SCF convergence fails non-deterministically.
5. Keep cutoffs low (8-15 Ry) and use Gamma-only for speed.
6. Full suite: `cd build && ctest --output-on-failure -E test_utils`


## 5. Build System Details

### CMake Options

| Option | Default | Description |
|--------|---------|-------------|
| `KRONOS_GPU_BACKEND` | `none` | GPU backend: `none`, `cuda`, `hip` |
| `KRONOS_BUILD_TESTS` | `ON` | Build GoogleTest suite |
| `KRONOS_BUILD_PYTHON` | `OFF` | Build pybind11 Python bindings |

### Dependencies

**Required:** FFTW3, BLAS, LAPACK, yaml-cpp.

**Optional** (with compile definitions): HDF5 (`KRONOS_HAS_HDF5`), MPI (`KRONOS_HAS_MPI`), libxc (`KRONOS_HAS_LIBXC`), spglib (`KRONOS_HAS_SPGLIB`).

**GPU:** CUDA toolkit defines `KRONOS_GPU_CUDA`; ROCm defines `KRONOS_GPU_HIP`.

### Build Commands

```bash
cmake -B build -S .                              # configure (CPU-only)
cmake --build build -j$(nproc)                   # compile
cd build && ctest --output-on-failure            # all tests
./build/src/kronos examples/si_bulk.yaml         # run (note: build/src/kronos)
./build/test_basis --gtest_filter='*BasisSize'   # single test
```


## 6. Coding Style and Conventions

### C++20 Features Used

- `std::numbers::pi`, `std::numbers::sqrt2` from `<numbers>`
- Structured bindings: `auto [crystal, input] = parse_input(file);`
- `constexpr` for all physical/math constants
- `[[nodiscard]]` on query methods and pure functions
- `std::map::contains()` (C++20)
- No compiler extensions (`CMAKE_CXX_EXTENSIONS OFF`)

### Precision

`double` and `std::complex<double>` only for physics. The aliases in `core/types.hpp` enforce this: `real_t = double`, `complex_t = std::complex<double>`, `CVec = std::vector<complex_t>`, `RVec = std::vector<double>`.

### Naming

- Classes: `PascalCase` -- `PlaneWaveBasis`, `SCFSolver`, `XCEvaluator`
- Functions: `snake_case` -- `compute_forces()`, `num_pw()`, `is_gga()`
- Constants: `snake_case` in `kronos::constants` -- `constants::pi`, `constants::rydberg_to_ev`
- Members: trailing underscore -- `crystal_`, `is_gga_`, `name_`
- Enums: `PascalCase` values -- `CalculationType::SCF`, `SmearingType::Gaussian`
- Macros: `KRONOS_UPPER_CASE` -- `KRONOS_TIMER`, `KRONOS_HAS_LIBXC`

### Unit System (Rydberg Atomic Units)

- Kinetic: `T = |k+G|^2` (not `/2` as in Hartree)
- Hartree: `V_H(G) = 8*pi*n(G)/|G|^2` (not `4*pi`)
- Coulomb tail: `V_loc(r) -> -2Z/r` (factor of 2 vs Hartree)
- Lattice: angstrom on input, bohr internally

### Profiling

Wrap expensive functions with `KRONOS_TIMER("name")`. The macro creates a `ScopedTimer` that records wall time in the global `TimerRegistry` singleton. Results appear in `SCFResult::timing` and JSON output.


## 7. Debugging Tips

### Structured JSON Logs

KRONOS emits structured JSON lines to stderr via `Logger::instance()`. Each line has `timestamp`, `level`, `event`, `message`, and custom fields. Capture with: `./build/src/kronos input.yaml 2>kronos.log`

```cpp
Logger::instance().info("scf_step", "SCF iteration complete",
    {{"step", std::to_string(step)}, {"energy_ry", std::to_string(energy)}});
```

### SCF Diagnostics

Each SCF step prints: step number, total energy, |dE|, density residual |dn|, wall time.

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| Energy oscillates | Mixing too aggressive | Reduce PulayMixer alpha (default 0.3) |
| Energy increases | Wrong sign in potential | Check V_loc(G=0), Hartree prefactor |
| Energy jump > 1 Ry | Numerical overflow or bug | Auto-aborts; check logs |
| Density residual stalls | Poor initial density | Check atomic density superposition |
| Davidson diverges | Near-degenerate states | Auto-switches to LOBPCG |

### Energy Decomposition

`SCFResult` provides: `kinetic_energy`, `hartree_energy`, `xc_energy`, `local_pp_energy`, `nonlocal_pp_energy`, `ewald_energy`, `smearing_energy`. Compare individual components against QE to isolate discrepancies. Ewald energy is the easiest to validate (no SCF dependence) -- if it matches QE to 5+ figures, the crystal setup is correct.

### Force Debugging

Validate forces via finite differences using `make_si_diamond_displaced(delta, atom_idx, dir)`:

```cpp
double F_numerical = -(E_plus - E_minus) / (2.0 * delta_bohr);
EXPECT_NEAR(F_numerical, F_analytic, FORCE_TOL);
```

Forces decompose into Ewald, local PP, and nonlocal PP components, all stored separately in `SCFResult`.

### Common Pitfalls

1. **Rydberg vs Hartree.** Factor-of-2 differences everywhere: kinetic `|k+G|^2` not `/2`, Hartree `8*pi` not `4*pi`, Coulomb `-2Z/r` not `-Z/r`.
2. **UPF conventions.** UPF stores `r*beta(r)` for projectors and `4*pi*r^2*rho(r)` for atomic density. Divide by appropriate powers of `r` when loading.
3. **G=0 terms.** Hartree sets `V_H(G=0) = 0`. Local PP has a finite `V_loc(G=0)`. Wrong sign shifts total energy by a constant.
4. **K-point formula.** KRONOS uses `k = (2n-N-1)/(2N) + s/(2N)`. For even N with shift=0, the grid is off-Gamma.
5. **FFT grid.** `ecutrho >= 4 * ecutwfc` for norm-conserving PPs. Grid dimensions come from `ecutrho`.


## 8. GPU Development

The `kronos::gpu` namespace is the sole vendor API boundary:

- `gpu/fft.hpp` -- cuFFT/rocFFT wrapper
- `gpu/blas.hpp` -- cuBLAS/rocBLAS wrapper
- `gpu/memory.hpp` -- device memory management
- `gpu_stubs.cpp` -- throws `GPUNotAvailableError` in CPU-only builds

**Adding a kernel:** (1) declare in `src/gpu/*.hpp`, (2) add stub in `gpu_stubs.cpp`, (3) implement in `.cu` (CUDA) or `.cpp` with `#ifdef KRONOS_GPU_HIP` (HIP), (4) call from physics code via `gpu::your_function()`. For deterministic results: `CUBLAS_WORKSPACE_CONFIG=:4096:8`.


## 9. Error Handling

| Situation | Action |
|-----------|--------|
| Invalid YAML input | `throw std::invalid_argument` with field name and allowed values |
| Unknown YAML key | Hard abort (strict schema) |
| UPF parse failure | `throw UPFParseError` with file path and line number |
| Negative density | Clamp to 0; abort if magnitude > 1e-6 |
| Energy oscillation > 1 Ry | Abort with diagnostic |
| SCF non-convergence | Return `SCFResult{converged=false}` (not an exception) |
| Davidson divergence | Auto-switch to LOBPCG |
| GPU OOM | Auto-fallback to CPU |

Exit codes from `main.cpp`: 0 (success), 1 (SCF not converged), 2 (input error), 3 (PP error), 4 (other).
