# KRONOS API Reference

All symbols reside in `namespace kronos` unless noted. Units are Rydberg atomic
units unless explicitly labeled. Lattice input is in angstrom (converted
internally to bohr).

---

## Core Module (`src/core/`)

### Type Aliases (`core/types.hpp`)

```cpp
using real_t    = double;                               // Always float64
using complex_t = std::complex<double>;                 // Always complex128
using Vec3      = std::array<double, 3>;                // 3D vector
using Mat3      = std::array<std::array<double, 3>, 3>; // 3x3 matrix (row-major)
using CVec      = std::vector<complex_t>;               // Complex wavefunction vector
using RVec      = std::vector<double>;                  // Real-space grid vector
```

### Enumerations (`core/types.hpp`)

```cpp
enum class CalculationType  { SCF, Relax, Bands, DOS };
enum class SmearingType     { None, Gaussian, MarzariVanderbilt, FermiDirac };
enum class EigensolverType  { Davidson, LOBPCG };
```

### Configuration Structs (`core/types.hpp`)

```cpp
struct KPointGrid {
    std::array<int, 3> grid{1, 1, 1};   // MP grid dimensions
    std::array<int, 3> shift{0, 0, 0};  // Shift flags (0 or 1)
};

struct Atom {
    std::string symbol;           // Element symbol ("Si", "Fe", ...)
    int         atomic_number{0}; // Z
    Vec3        position{};       // Fractional coordinates in [0, 1)
};

struct CalculationParams {
    CalculationType type{CalculationType::SCF};
    double          ecutwfc{30.0};            // Plane-wave cutoff (Ry)
    double          ecutrho{0.0};             // Density cutoff (0 = 4x ecutwfc)
    KPointGrid      kpoints{};
    std::string     xc_functional{"LDA_PZ"};
    SmearingType    smearing{SmearingType::None};
    double          degauss{0.01};            // Smearing width (Ry)
    bool            spin_polarized{false};
    EigensolverType eigensolver{EigensolverType::Davidson};
};

struct ConvergenceParams {
    double energy_threshold{1e-8};   // Ry
    double density_threshold{1e-9};
    int    max_scf_steps{100};
    double force_threshold{1e-3};    // Ry/bohr
};

struct HardwareParams {
    bool        use_gpu{false};
    std::string gpu_backend{"none"};  // "cuda", "hip", "none"
    int         mpi_tasks{1};
};

struct InputData {
    CalculationParams                  calculation;
    ConvergenceParams                  convergence;
    HardwareParams                     hardware;
    std::map<std::string, std::string> pseudopotentials; // symbol -> filepath
};
```

### Crystal (`core/crystal.hpp`)

Lattice vectors + atomic basis. Stores lattice in angstrom; caches reciprocal
lattice and volume in atomic units on construction.

| Constructor | Description |
|-------------|-------------|
| `Crystal()` | Default (empty). |
| `Crystal(const Mat3& lattice_angstrom, std::vector<Atom> atoms)` | Build from lattice (angstrom, row vectors) and atoms. Throws `std::invalid_argument` if degenerate or empty. |

| Method | Return | Description |
|--------|--------|-------------|
| `lattice()` | `const Mat3&` | Lattice vectors in angstrom. |
| `lattice_bohr()` | `Mat3` | Lattice vectors in bohr. |
| `reciprocal_lattice()` | `Mat3` | Reciprocal lattice (2*pi/a) in 1/bohr. |
| `volume()` | `double` | Cell volume in bohr^3. |
| `num_atoms()` | `size_t` | Atom count. |
| `atoms()` | `const std::vector<Atom>&` | Full atom list. |
| `atom(size_t i)` | `const Atom&` | i-th atom (bounds-checked). |
| `total_electrons()` | `int` | Sum of atomic_number values. |
| `frac_to_cart(frac)` | `Vec3` | Fractional to Cartesian (bohr). |
| `cart_to_frac(cart)` | `Vec3` | Cartesian (bohr) to fractional. |

### Element Data (`core/element_data.hpp`)

| Function | Description |
|----------|-------------|
| `int atomic_number_from_symbol(symbol)` | Z from symbol. Throws `std::invalid_argument` if unknown. Range: Z=1..86. |
| `std::string symbol_from_atomic_number(z)` | Symbol from Z. Throws `std::out_of_range` if outside [1, 86]. |

### Spherical Harmonics (`core/spherical_harmonics.hpp`)

```cpp
double real_spherical_harmonic(int l, int m, double x, double y, double z);
```

Real spherical harmonic Y_lm at direction (x, y, z). Normalizes internally.
Supported: l=0..3, m=-l..+l. QE convention. Returns 0 for unsupported (l, m).

### Constants (`core/constants.hpp`)

Namespace: `kronos::constants`. Key values: `bohr_to_angstrom` (0.529177),
`rydberg_to_ev` (13.6057), `hartree_to_ev` (27.2114), `pi`, `two_pi`,
`four_pi`, `kboltzmann_ry_per_K` (6.334e-6).

---

## Basis Module (`src/basis/`)

### PlaneWaveBasis (`basis/plane_wave.hpp`)

```cpp
struct GVector {
    int h, k, l;        // Miller indices
    Vec3 cart;           // Cartesian (1/bohr)
    double norm2;        // |G|^2
};
```

| Constructor | Description |
|-------------|-------------|
| `PlaneWaveBasis(crystal, ecutwfc, k_max=0.0)` | Enumerate G with |G|^2 <= ecutwfc. When k_max>0, expands sphere for all k-points with |k|<=k_max. |

| Method | Return | Description |
|--------|--------|-------------|
| `num_pw()` | `size_t` | Number of plane waves. |
| `gvectors()` | `const std::vector<GVector>&` | All G-vectors. |
| `gvec(i)` | `const GVector&` | i-th G-vector. |
| `kinetic_energies(k_frac)` | `std::vector<double>` | |k+G|^2 for given k-point (Ry, NOT /2). |
| `ecutwfc()` | `double` | Energy cutoff (Ry). |
| `max_miller()` | `std::array<int,3>` | Max absolute Miller index per direction. |

### FFTGrid (`basis/fft_grid.hpp`)

FFTW3-backed 3D FFT. Grid dims are FFT-friendly (products of 2, 3, 5).
Non-copyable, move-constructible.

| Constructor | Description |
|-------------|-------------|
| `FFTGrid(basis, ecutrho)` | Grid for explicit density cutoff. |
| `FFTGrid(basis)` | ecutrho = 4 * ecutwfc. |

| Method | Return | Description |
|--------|--------|-------------|
| `dims()` | `std::array<int,3>` | {n1, n2, n3}. |
| `total_points()` | `int` | n1*n2*n3. |
| `forward(r_space, g_space)` | `void` | FFT: real-space to G-space. |
| `inverse(g_space, r_space)` | `void` | IFFT: G-space to real-space. |
| `gvec_to_index(h, k, l)` | `int` | Miller indices to linear FFT index. |
| `scatter_to_grid(basis, pw_coeffs, grid)` | `void` | PW coefficients onto FFT grid. |
| `gather_from_grid(basis, grid, pw_coeffs)` | `void` | Extract PW coefficients from FFT grid. |

### KPointGenerator (`basis/kpoints.hpp`)

```cpp
struct KPointData {
    std::vector<Vec3>   kpoints;  // Fractional reciprocal coords
    std::vector<double> weights;  // Sum to 1.0
};
```

```cpp
static KPointData KPointGenerator::generate_monkhorst_pack(grid, crystal);
```

Grid formula: k_i = (2n_i - N_i - 1)/(2N_i) + shift_i/(2N_i). Time-reversal
symmetry folding applied (k and -k merged, weight doubled).

---

## IO Module (`src/io/`)

### UPF Parser (`io/upf_parser.hpp`)

```cpp
struct RadialGrid { int npoints; std::vector<double> r, rab; };
struct BetaProjector { int index, angular_momentum, cutoff_index; std::vector<double> values; };
struct AtomicWavefunction { int angular_momentum; double occupation; std::string label; std::vector<double> values; };

struct PseudoPotential {
    std::string element;  int atomic_number;  double z_valence;
    std::string pp_type;  // "NC", "US", "PAW"
    bool is_norm_conserving, is_ultrasoft, is_paw;
    std::string xc_functional;
    double total_psenergy, wfc_cutoff, rho_cutoff;
    int lmax, num_projectors, num_wfc;
    RadialGrid mesh;
    std::vector<double> vloc;                     // V_loc(r) in Ry
    std::vector<BetaProjector> betas;
    std::vector<std::vector<double>> dij;         // D_ij [nproj x nproj]
    std::vector<double> rho_atomic;
    std::vector<AtomicWavefunction> atomic_wfc;
};
```

| Function | Description |
|----------|-------------|
| `PseudoPotential parse_upf(filepath)` | Parse UPF v2 file. Throws `UPFParseError`. |
| `void validate_pseudopotential(pp)` | Norm-conservation check, z_valence > 0. Mandatory on load. |

### Input Parser (`io/input_parser.hpp`)

```cpp
struct ParsedInput { Crystal crystal; InputData input; };
```

| Function | Description |
|----------|-------------|
| `ParsedInput parse_input(filepath)` | Parse YAML with strict schema. Throws `InputValidationError`. |
| `ParsedInput parse_input_string(yaml)` | Parse from string (for tests). |

### Output Writer (`io/output_writer.hpp`)

| Method | Description |
|--------|-------------|
| `OutputWriter::write_json(filepath, result, crystal, calc_type)` | Atomic write (temp + rename). |
| `OutputWriter::to_json_string(result, crystal, calc_type)` | Serialize to JSON string. |

---

## Potential Module (`src/potential/`)

### HartreeSolver (`potential/hartree.hpp`)

V_H(G) = 8*pi*n(G)/|G|^2 for G!=0; V_H(G=0) = 0.

| Constructor | `explicit HartreeSolver(const PlaneWaveBasis& basis)` |
|-------------|-------------------------------------------------------|

| Method | Return | Description |
|--------|--------|-------------|
| `compute(density_g)` | `CVec` | Hartree potential in G-space. |
| `energy(density_g, vhartree_g, volume, num_grid)` | `double` | E_H = (Omega/2) sum_G conj(V_H)*n (Ry). |

### XCEvaluator (`potential/xc.hpp`)

Wraps libxc or built-in LDA-PZ fallback. Non-copyable.

Supported: `"LDA_PZ"`, `"LDA_PW"` (LDA), `"PBE"`, `"PBEsol"` (GGA).

```cpp
struct XCResult { RVec exc, vxc, vsigma; double energy; };
```

| Method | Return | Description |
|--------|--------|-------------|
| `evaluate(density_r, volume)` | `XCResult` | LDA evaluation. |
| `evaluate_gga(density_r, sigma_r, volume)` | `XCResult` | GGA. sigma_r = |nabla n|^2. |
| `is_gga()` | `bool` | True if functional needs gradients. |
| `name()` | `const std::string&` | Functional name. |

### LocalPPEvaluator (`potential/local_pp.hpp`)

V_loc(G) = sum_species V_loc^s(|G|) * S_s(G). Uses Coulomb tail subtraction.

| Method | Return | Description |
|--------|--------|-------------|
| `vloc_g()` | `const CVec&` | Precomputed V_loc(G) (Ry). |
| `energy(density_g, volume, num_grid)` | `double` | E_loc = Omega * sum Re[conj(V_loc)*n]. |
| `vloc_of_q(pp, q, volume)` | `double` | (static) Radial FT of V_loc at |G|=q. |
| `structure_factor(positions, g_cart)` | `complex_t` | (static) S(G) = sum exp(-iG.tau). |

### NonlocalPP (`potential/nonlocal_pp.hpp`)

Kleinman-Bylander form: V_NL = sum D_ij |beta_i><beta_j|. Each UPF projector
with angular momentum l expands to (2l+1) m-channels.

| Method | Return | Description |
|--------|--------|-------------|
| `prepare_kpoint(k_frac)` | `void` | Cache beta projectors for this k-point. |
| `apply(psi_g, k_frac)` | `CVec` | V_NL|psi>. Uses cache if available. |
| `energy(wfns, occupations, k_frac)` | `double` | E_NL = sum f_n <psi|V_NL|psi>. |
| `num_projectors()` | `int` | Total expanded projector count. |

### EwaldCalculator (`potential/ewald.hpp`)

Ion-ion Ewald summation: E_ion = E_real + E_recip + E_self + E_charged.

```cpp
struct EwaldCalculator::Result { double energy; std::vector<Vec3> forces; };
```

| Method | Description |
|--------|-------------|
| `Result compute(crystal, charges)` | (static) From explicit valence charges. |
| `Result compute(crystal, pseudopotentials)` | (static) Extract z_valence, then compute. |

### ForceCalculator (`potential/forces.hpp`)

F_I = F_ewald + F_local + F_nonlocal (all in Ry/bohr).

| Method | Return | Description |
|--------|--------|-------------|
| `compute_local_forces(crystal, basis, pps, density_g, num_grid)` | `vector<Vec3>` | dE_loc/dR via structure factor derivative. |
| `compute_nonlocal_forces(crystal, basis, pps, wfns, occs, kpts, kwts, spin_fac)` | `vector<Vec3>` | dE_NL/dR via KB projector derivatives. |
| `compute_total_forces(ewald, local, nonlocal)` | `vector<Vec3>` | Element-wise sum. |

### GGA Gradients (`potential/gradient.hpp`)

| Function | Return | Description |
|----------|--------|-------------|
| `compute_sigma(density_g, basis, fft_grid)` | `RVec` | |nabla n(r)|^2 via G-space differentiation. |
| `compute_gga_potential(density_g, vsigma, basis, fft_grid)` | `RVec` | V_gga(r) = -2 div(vsigma * nabla n). |

---

## Hamiltonian Module (`src/hamiltonian/`)

### Hamiltonian (`hamiltonian/hamiltonian.hpp`)

H|psi> = T|psi> + V_eff|psi> + V_NL|psi>. Kinetic in G-space, local potential
via FFT, nonlocal via projector overlaps.

| Constructor | `Hamiltonian(crystal, basis, fft_grid, nonlocal_pp)` |
|-------------|------------------------------------------------------|

| Method | Return | Description |
|--------|--------|-------------|
| `update_veff(veff_r)` | `void` | Set effective potential on real-space grid (each SCF step). |
| `apply(psi_g, k_frac)` | `CVec` | H|psi> in G-space. |
| `get_apply_function(k_frac)` | `function<CVec(CVec)>` | Callable for Davidson. Caches nonlocal projectors. |
| `kinetic_diagonal(k_frac)` | `vector<double>` | |k+G|^2 for preconditioner. |

---

## Solver Module (`src/solver/`)

### SCFSolver (`solver/scf.hpp`)

```cpp
struct SCFResult {
    bool converged;  int scf_steps;
    double total_energy_ry, total_energy_ev, fermi_energy_ev;
    double kinetic_energy, hartree_energy, xc_energy;
    double local_pp_energy, nonlocal_pp_energy, ewald_energy, smearing_energy;
    std::vector<Vec3> forces, ewald_forces, local_forces, nonlocal_forces;
    std::vector<std::vector<double>> eigenvalues;  // [k][band] Ry
    std::vector<complex_t> converged_veff_r;
    std::map<std::string, double> timing;
};
```

| Constructor | `SCFSolver(crystal, calc_params, conv_params, pseudopotentials)` |
|-------------|------------------------------------------------------------------|

| Method | Return | Description |
|--------|--------|-------------|
| `solve()` | `SCFResult` | Run full SCF loop to convergence. |

### DavidsonSolver (`solver/davidson.hpp`)

```cpp
struct EigenResult {
    std::vector<double> eigenvalues;  std::vector<CVec> eigenvectors;
    int iterations;  bool converged;  double max_residual;
};
```

| Params field | Default | Description |
|-------------|---------|-------------|
| `max_iterations` | 100 | Max Davidson iterations. |
| `tolerance` | 1e-6 | Residual norm threshold. |
| `subspace_factor` | 3 | Subspace = factor * num_bands. |
| `max_subspace` | 0 | 0 = auto (3 * num_bands). |

| Method | Return | Description |
|--------|--------|-------------|
| `solve(h_apply, preconditioner, num_bands, num_pw, initial_guess)` | `EigenResult` | Find lowest num_bands eigenvalues. |

### FermiSolver (`solver/fermi.hpp`)

```cpp
struct FermiResult {
    double fermi_energy;  // Ry
    std::vector<std::vector<double>> occupations;  // [k][band]
    double total_electrons_found;  bool converged;
};
```

```cpp
static FermiResult find_fermi_level(eigenvalues, weights, target_electrons,
                                     smearing, degauss, spin_factor=2);
```

### Mixing (`solver/mixing.hpp`)

| Class | Constructor | Key Method |
|-------|------------|------------|
| `LinearMixer` | `LinearMixer(alpha=0.3)` | `RVec mix(n_in, n_out)` -- n_new = alpha*n_out + (1-alpha)*n_in |
| `PulayMixer` | `PulayMixer(max_history=8, alpha=0.3)` | `RVec mix(n_in, n_out)` -- DIIS minimization. Also: `reset()`, `history_size()`. |
| `KerkerPreconditioner` | `KerkerPreconditioner(q0=1.5)` | `CVec apply(residual_g, g_norm2)` -- R(G)*|G|^2/(|G|^2+q0^2) |

### BFGSOptimizer (`solver/bfgs.hpp`)

```cpp
struct RelaxResult {
    bool converged;  int relax_steps;
    double final_energy_ry, final_energy_ev, max_force_ry_bohr;
    Crystal final_crystal;  SCFResult final_scf;
    std::vector<double> energy_history, force_history;
};
```

| Params field | Default | Description |
|-------------|---------|-------------|
| `max_steps` | 50 | Max ionic steps. |
| `force_threshold` | 1e-3 | Ry/bohr. |
| `energy_threshold` | 1e-6 | Ry. |
| `initial_step` | 0.5 | bohr (trust radius). |
| `max_step` | 1.0 | bohr (max displacement). |

| Method | Return | Description |
|--------|--------|-------------|
| `optimize(crystal, calc_params, conv_params, pps)` | `RelaxResult` | Run BFGS geometry relaxation. |

---

## PostProcessing Module (`src/postprocessing/`)

### BandStructureCalculator (`postprocessing/band_structure.hpp`)

```cpp
struct HighSymmetryPoint { std::string label; Vec3 frac; };
using KPathSpec = std::vector<HighSymmetryPoint>;
struct BandData {
    std::vector<Vec3> kpoints;  std::vector<double> distances;
    std::vector<std::vector<double>> eigenvalues;  // [nk][nbands] Ry
    std::vector<std::pair<double, std::string>> tick_positions;
};
```

| Method | Description |
|--------|-------------|
| `generate_kpath(crystal, path_spec, npts=50)` | Interpolated k-path from high-symmetry points. |
| `compute_bands(kpath, h_apply_factory, precond_factory, nbands, npw_func)` | Diagonalize at each k. Fills eigenvalues in-place. |
| `write_bands_gnuplot(filename, band_data)` | Columns: k_dist, band1_eV, band2_eV, ... |
| `default_path_fcc/bcc/sc/hcp()` | Predefined high-symmetry paths. |

### DOSCalculator (`postprocessing/dos.hpp`)

```cpp
struct DOSData { std::vector<double> energies, dos_values, integrated_dos; };
```

| Method | Description |
|--------|-------------|
| `compute_dos(eigenvalues, weights, smearing, degauss=0.05, emin=-20, emax=20, npts=2001, spin_factor=2)` | Smeared DOS. Eigenvalues in Ry, degauss/energy range in eV. |
| `write_dos(filename, dos_data)` | Columns: energy(eV), dos(states/eV), integrated. |

---

## Utils Module (`src/utils/`)

### Timer (`utils/timer.hpp`)

```cpp
struct TimingEntry { std::string name; double total_seconds; int call_count; };
```

**TimerRegistry** (singleton): `instance()`, `record(name, secs)`, `entries()`,
`reset()`, `print_summary()`, `as_map()`. Thread-safe.

**ScopedTimer** (RAII): `ScopedTimer(name)` -- records elapsed time on destruction.

**Macro:** `KRONOS_TIMER("label")` -- creates a scoped timer for the enclosing block.

### Logger (`utils/logger.hpp`)

`enum class LogLevel { Debug, Info, Warning, Error };`

**Logger** (singleton): `instance()`, `set_level(level)`, `set_mpi_rank(rank)`,
`log(level, event, message, fields)`, plus `debug/info/warning/error` convenience
methods. Outputs structured JSON lines to stderr.

### Radial Integration (`utils/radial_integral.hpp`)

```cpp
double simpson_radial(const double* func, const double* rab, int npts);
double simpson_radial(const std::vector<double>& func, const std::vector<double>& rab, int npts);
```

Simpson's rule (1-4-2-4-...-4-1 pattern) matching QE convention. Trapezoidal
fallback for even point count.

---

## GPU Abstraction Layer (`src/gpu/`)

Namespace `kronos::gpu`. v0.1 stubs throw `GPUNotAvailableError`.

**GPUFFTGrid** (`gpu/fft.hpp`): `GPUFFTGrid(dims)`, `forward()`, `inverse()`, `dims()`.

**BLAS** (`gpu/blas.hpp`): `gemm(m,n,k,alpha,A,lda,B,ldb,beta,C,ldc)`, `zdotc(n,x,y)`.

**Memory** (`gpu/memory.hpp`): `gpu_malloc`, `gpu_free`, `gpu_memcpy_h2d/d2h/d2d`,
`gpu_available()`, `gpu_memory_free()`, `gpu_memory_total()`.
