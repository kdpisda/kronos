# KRONOS Developer Guide

## Directory Layout

```
kronos/
├── src/
│   ├── core/           # Fundamental types, constants, Crystal class
│   ├── basis/          # Plane-wave basis and FFT grid
│   ├── io/             # Input parsing (YAML), output (JSON/HDF5), UPF reader
│   ├── potential/      # All potential terms: Hartree, XC, local/nonlocal PP, Ewald
│   ├── solver/         # SCF loop, Davidson, mixing, Fermi solver, BFGS
│   ├── hamiltonian/    # H|ψ⟩ application (the computational kernel)
│   ├── postprocessing/ # Band structure, DOS
│   ├── gpu/            # GPU abstraction layer (CUDA/HIP stubs for CPU builds)
│   └── utils/          # Timer, logger
├── test/               # GoogleTest test suite
├── examples/           # Example input files
├── docs/               # Documentation
└── CMakeLists.txt      # Top-level build
```

Each module in `src/` is a directory containing `.hpp` (public API) and `.cpp` (implementation) files. Headers are included via module-relative paths: `#include "core/types.hpp"`.

---

## How to Add a New XC Functional

1. **Check libxc support**: If the functional exists in libxc, it may already work. Check `XCEvaluator::XCEvaluator()` in `src/potential/xc.cpp` for the name-to-libxc-id mapping.

2. **Add name mapping** in `xc.cpp`:
   ```cpp
   // In the constructor's name resolution:
   if (name == "YOUR_FUNC") {
       // Set exchange and correlation IDs from libxc
       xc_id = XC_GGA_X_YOUR_FUNC;  // or LDA_X_...
       correlation_id = XC_GGA_C_YOUR_FUNC;
   }
   ```

3. **Handle LDA vs GGA**: If it's a GGA functional, ensure `is_gga()` returns `true` so the SCF loop computes |∇ρ|² via `GGAGradient` before calling `evaluate_gga()`.

4. **Add built-in fallback** (optional): If you want the functional to work without libxc, add the analytic expressions in `xc.cpp`'s built-in path.

5. **Add input validation** in `src/io/input_parser.cpp`: Add the new name to the allowed values for `xc_functional`.

6. **Add tests** in `test/test_scf.cpp` or `test/test_physics.cpp`:
   ```cpp
   TEST(XCFunctional, YourFuncProperties) {
       XCEvaluator xc("YOUR_FUNC");
       // Test energy, potential at known densities
   }
   ```

---

## How to Add a New Potential Term

1. **Create files**: `src/potential/your_potential.hpp` and `.cpp`

2. **Define the class**:
   ```cpp
   class YourPotential {
   public:
       YourPotential(const Crystal& crystal, const PlaneWaveBasis& basis, ...);
       CVec compute(const CVec& density_g) const;  // V(G) in G-space
       double energy(const CVec& density_g, ...) const;  // Energy in Ry
   };
   ```

3. **Integrate into SCF** in `src/solver/scf.cpp`:
   - Construct the potential object in `SCFSolver::SCFSolver()`
   - Call `compute()` in the SCF loop to get V(G)
   - Add V(G) to `V_eff`
   - Call `energy()` and add to `SCFResult`

4. **Add to CMakeLists.txt**: The file is automatically picked up if it's in `src/potential/`.

5. **Write tests**: Add a test file or extend `test/test_physics.cpp`.

---

## Test Architecture

### Test Files

| File | What it tests |
|------|---------------|
| `test_input.cpp` | YAML parsing, validation, error handling |
| `test_basis.cpp` | PlaneWaveBasis, FFTGrid properties |
| `test_fft.cpp` | FFT round-trip, scatter/gather |
| `test_upf.cpp` | UPF parsing, validation |
| `test_crystal.cpp` | Crystal construction, coordinate transforms |
| `test_hamiltonian.cpp` | H|ψ⟩ properties: linearity, Hermiticity, kinetic |
| `test_solvers.cpp` | Davidson, mixing, Fermi solver |
| `test_scf.cpp` | SCF convergence, energy components |
| `test_physics.cpp` | Ewald, XC functionals, Hartree, k-points |
| `test_forces.cpp` | Force computation, Newton's 3rd law, BFGS |
| `test_gradient.cpp` | GGA gradient computation |
| `test_postprocessing.cpp` | Band structure, DOS |
| `test_output.cpp` | JSON/HDF5 output |
| `test_utils.cpp` | Timer, logger |
| `test_validation.cpp` | Physics invariant validation |
| `test_convergence.cpp` | Cutoff/k-point convergence studies |
| `test_regression.cpp` | Frozen baseline regression |

### Test Helpers

`test/test_helpers.hpp` provides factory functions in `namespace kronos::test`:

- `make_si_diamond_crystal()` — Standard Si diamond cell
- `make_nacl_crystal()` — NaCl rocksalt 8-atom cell
- `make_cscl_crystal()` — CsCl 2-atom cell
- `make_si_pseudopotential()` — Gaussian local-only Si PP
- `make_si_pseudopotential_nonlocal()` — Si PP with KB projector
- `make_si_pp_map()` / `make_si_pp_map_nonlocal()` — PP map wrappers
- `make_si_diamond_displaced()` — Displaced crystal for force tests
- `make_h_pseudopotential()` — Hydrogen PP

### Conventions

- Tests that require SCF convergence use `GTEST_SKIP()` if convergence fails
- Tolerance constants are defined in `test_helpers.hpp` (ENERGY_TOL, FORCE_TOL, etc.)
- Each test file includes only the headers it needs
- SCF tests use low cutoffs (8-15 Ry) and Gamma-only for speed

### Adding a New Test

1. Create `test/test_yourmodule.cpp`
2. Add `kronos_add_test(test_yourmodule)` to `test/CMakeLists.txt`
3. Build and run: `cmake --build build && ./build/test/test_yourmodule`

---

## Build System

### CMake Structure

```
CMakeLists.txt          # Top-level: project, options, find_package, add_subdirectory
├── src/CMakeLists.txt  # Builds kronos_lib (static library)
└── test/CMakeLists.txt # Fetches GoogleTest, builds test executables
```

The `kronos_lib` target contains all source files. Test executables link against `kronos_lib` and `GTest::gtest_main`.

### Adding a New Source File

Just add `.cpp` to `src/your_module/`. CMake globs or lists are in `src/CMakeLists.txt`.

### Adding a Dependency

In the top-level `CMakeLists.txt`:
```cmake
find_package(YourLib REQUIRED)
target_link_libraries(kronos_lib PUBLIC YourLib::YourLib)
```

### GPU Backends

Set `-DKRONOS_GPU_BACKEND=cuda` or `hip`. This:
1. Enables CUDA/HIP language in CMake
2. Links cuFFT/rocFFT, cuBLAS/rocBLAS
3. Compiles `src/gpu/` with actual GPU implementations instead of CPU stubs

The `gpu::` namespace is the abstraction boundary. Physics code never calls CUDA/HIP directly.

---

## Coding Conventions

### Language

- **C++20** standard (concepts, ranges, `std::numbers`)
- All floating-point: `double` (never `float` for physics)
- Complex: `std::complex<double>`

### Naming

- Classes: `PascalCase` (`PlaneWaveBasis`, `SCFSolver`)
- Functions/methods: `snake_case` (`compute_forces`, `num_pw`)
- Constants: `snake_case` in `constants::` namespace
- Member variables: `snake_case_` with trailing underscore
- Enums: `PascalCase` values (`CalculationType::SCF`)

### Error Handling

- Input validation: `throw std::invalid_argument` with descriptive message
- File I/O errors: `throw std::runtime_error` with file path
- UPF errors: `throw UPFParseError` with line number
- SCF non-convergence: return `SCFResult` with `converged=false` (not an exception)
- Numerical issues (NaN, overflow): abort with diagnostic message

### Headers

- Every header has `#pragma once`
- Include what you use; no transitive dependency assumptions
- System headers after project headers
- Forward-declare where possible to reduce compile times

---

## Profiling

### KRONOS_TIMER

The built-in timer in `src/utils/timer.hpp`:

```cpp
#include "utils/timer.hpp"

void my_function() {
    KRONOS_TIMER("my_function");  // Scoped timer
    // ... work ...
}  // Timer stops here, records wall time
```

Results appear in `SCFResult::timing` map and in JSON log output.

### GPU Profiling

For CUDA builds, NVTX ranges are automatically inserted around GPU kernels. Profile with:

```bash
nsys profile ./kronos input.yaml
nsys stats report.nsys-rep
```
