# KRONOS API Reference

## Core Module (`src/core/`)

### Types (`core/types.hpp`)

```cpp
namespace kronos {
    using real_t    = double;
    using complex_t = std::complex<double>;
    using Vec3      = std::array<double, 3>;
    using Mat3      = std::array<std::array<double, 3>, 3>;
    using CVec      = std::vector<complex_t>;
    using RVec      = std::vector<double>;

    enum class CalculationType  { SCF, Relax, Bands, DOS };
    enum class SmearingType     { None, Gaussian, MarzariVanderbilt, FermiDirac };
    enum class EigensolverType  { Davidson, LOBPCG };
}
```

### Atom

```cpp
struct Atom {
    std::string symbol;       // Element symbol ("Si", "Fe", ...)
    int         atomic_number; // Z
    Vec3        position;      // Fractional coordinates in [0,1)
};
```

### CalculationParams

```cpp
struct CalculationParams {
    CalculationType type{SCF};
    double          ecutwfc{30.0};        // Plane-wave cutoff (Ry)
    double          ecutrho{0.0};         // Density cutoff (0=auto)
    KPointGrid      kpoints{};
    std::string     xc_functional{"LDA_PZ"};
    SmearingType    smearing{None};
    double          degauss{0.01};
    bool            spin_polarized{false};
    EigensolverType eigensolver{Davidson};
};
```

### ConvergenceParams

```cpp
struct ConvergenceParams {
    double energy_threshold{1e-8};     // Ry
    double density_threshold{1e-9};
    int    max_scf_steps{100};
    double force_threshold{1e-3};      // Ry/bohr
};
```

### Crystal (`core/crystal.hpp`)

```cpp
class Crystal {
public:
    Crystal(const Mat3& lattice_angstrom, std::vector<Atom> atoms);

    const Mat3& lattice() const;            // Angstrom
    Mat3 lattice_bohr() const;              // bohr
    Mat3 reciprocal_lattice() const;        // 2π/a convention, 1/bohr
    double volume() const;                  // bohr³
    size_t num_atoms() const;
    const std::vector<Atom>& atoms() const;
    const Atom& atom(size_t i) const;       // Bounds-checked
    int total_electrons() const;            // Sum of atomic_number
    Vec3 frac_to_cart(const Vec3& frac) const;  // → bohr
    Vec3 cart_to_frac(const Vec3& cart) const;  // cart in bohr → frac
};
```

**Example:**
```cpp
Mat3 lattice = {{{0, 2.715, 2.715}, {2.715, 0, 2.715}, {2.715, 2.715, 0}}};
std::vector<Atom> atoms = {{"Si", 14, {0, 0, 0}}, {"Si", 14, {0.25, 0.25, 0.25}}};
Crystal si(lattice, std::move(atoms));
// si.volume() → cell volume in bohr³
// si.num_atoms() → 2
```

---

## Basis Module (`src/basis/`)

### GVector

```cpp
struct GVector {
    int h, k, l;       // Miller indices
    Vec3 cart;          // Cartesian coordinates (1/bohr)
    double norm2;       // |G|²
};
```

### PlaneWaveBasis (`basis/plane_wave.hpp`)

```cpp
class PlaneWaveBasis {
public:
    PlaneWaveBasis(const Crystal& crystal, double ecutwfc);

    size_t num_pw() const;                      // Number of plane waves
    const std::vector<GVector>& gvectors() const;
    const GVector& gvec(size_t i) const;
    double ecutwfc() const;                     // Stored cutoff (Ry)
    std::array<int, 3> max_miller() const;      // Max |h|, |k|, |l|

    // Kinetic energies |k+G|²/2 for a given k-point (fractional recip coords)
    std::vector<double> kinetic_energies(const Vec3& k_frac) const;
};
```

**Example:**
```cpp
PlaneWaveBasis basis(crystal, 30.0);  // 30 Ry cutoff
auto ke = basis.kinetic_energies({0, 0, 0});  // Gamma-point kinetic energies
```

### FFTGrid (`basis/fft_grid.hpp`)

```cpp
class FFTGrid {
public:
    FFTGrid(const PlaneWaveBasis& basis, double ecutrho);
    explicit FFTGrid(const PlaneWaveBasis& basis);  // ecutrho = 4×ecutwfc

    std::array<int, 3> dims() const;      // {n1, n2, n3}
    int total_points() const;             // n1×n2×n3

    void forward(const std::vector<complex_t>& r_space,
                 std::vector<complex_t>& g_space);
    void inverse(const std::vector<complex_t>& g_space,
                 std::vector<complex_t>& r_space);

    int gvec_to_index(int h, int k, int l) const;

    void scatter_to_grid(const PlaneWaveBasis& basis,
                         const CVec& pw_coeffs,
                         std::vector<complex_t>& grid) const;
    void gather_from_grid(const PlaneWaveBasis& basis,
                          const std::vector<complex_t>& grid,
                          CVec& pw_coeffs) const;
};
```

---

## IO Module (`src/io/`)

### PseudoPotential (`io/upf_parser.hpp`)

```cpp
struct RadialGrid {
    int npoints;
    std::vector<double> r;      // Radial grid points
    std::vector<double> rab;    // Integration weights
};

struct BetaProjector {
    int index;
    int angular_momentum;       // l quantum number
    int cutoff_index;
    std::vector<double> values; // β(r)
};

struct PseudoPotential {
    std::string element;
    int         atomic_number;
    double      z_valence;
    std::string pp_type;         // "NC", "US", "PAW"
    bool        is_norm_conserving, is_ultrasoft, is_paw;
    std::string xc_functional;
    int         lmax;
    int         num_projectors;

    RadialGrid                       mesh;
    std::vector<double>              vloc;       // V_loc(r) in Ry
    std::vector<BetaProjector>       betas;
    std::vector<std::vector<double>> dij;        // D_ij matrix
    std::vector<double>              rho_atomic;
};

PseudoPotential parse_upf(const std::string& filepath);
void validate_pseudopotential(const PseudoPotential& pp);
```

---

## Potential Module (`src/potential/`)

### HartreeSolver (`potential/hartree.hpp`)

```cpp
class HartreeSolver {
public:
    explicit HartreeSolver(const PlaneWaveBasis& basis);

    CVec compute(const CVec& density_g) const;
    // V_H(G) = 8π·n(G)/|G|²; V_H(G=0) = 0

    double energy(const CVec& density_g, const CVec& vhartree_g,
                  double volume, int num_grid) const;
    // E_H = (Ω/2) Σ_G conj(V_H(G))·n(G)
};
```

### XCEvaluator (`potential/xc.hpp`)

```cpp
struct XCResult {
    RVec   exc;     // Energy density per grid point (Ry)
    RVec   vxc;     // V_xc(r) per grid point (Ry)
    RVec   vsigma;  // dE/d(σ) per grid point (GGA only)
    double energy;  // Total XC energy (Ry)
};

class XCEvaluator {
public:
    explicit XCEvaluator(const std::string& functional_name);

    XCResult evaluate(const RVec& density_r, double cell_volume) const;
    XCResult evaluate_gga(const RVec& density_r, const RVec& sigma_r,
                          double cell_volume) const;
    bool is_gga() const;
    const std::string& name() const;
};
```

### LocalPPEvaluator (`potential/local_pp.hpp`)

```cpp
class LocalPPEvaluator {
public:
    LocalPPEvaluator(const Crystal& crystal, const PlaneWaveBasis& basis,
                     const std::map<std::string, PseudoPotential>& pseudopotentials);

    const CVec& vloc_g() const;    // Precomputed V_loc(G) (Ry)
    double energy(const CVec& density_g, double volume, int num_grid) const;
};
```

### NonlocalPP (`potential/nonlocal_pp.hpp`)

```cpp
class NonlocalPP {
public:
    NonlocalPP(const Crystal& crystal, const PlaneWaveBasis& basis,
               const std::map<std::string, PseudoPotential>& pseudopotentials);

    CVec apply(const CVec& psi_g, const Vec3& k_frac) const;
    // V_NL|ψ⟩ = Σ D_l |β⟩⟨β|ψ⟩

    double energy(const std::vector<CVec>& wavefunctions,
                  const std::vector<double>& occupations,
                  const Vec3& k_frac) const;

    int num_projectors() const;
};
```

### EwaldCalculator (`potential/ewald.hpp`)

```cpp
class EwaldCalculator {
public:
    struct Result {
        double energy;              // Ion-ion energy (Ry)
        std::vector<Vec3> forces;   // Force per atom (Ry/bohr)
    };

    static Result compute(const Crystal& crystal,
                          const std::vector<double>& charges);
    static Result compute(const Crystal& crystal,
                          const std::map<std::string, PseudoPotential>& pps);
};
```

### ForceCalculator (`potential/forces.hpp`)

```cpp
class ForceCalculator {
public:
    static std::vector<Vec3> compute_local_forces(
        const Crystal&, const PlaneWaveBasis&,
        const std::map<std::string, PseudoPotential>&,
        const CVec& density_g, int num_grid);

    static std::vector<Vec3> compute_nonlocal_forces(
        const Crystal&, const PlaneWaveBasis&,
        const std::map<std::string, PseudoPotential>&,
        const std::vector<std::vector<CVec>>& wavefunctions,
        const std::vector<std::vector<double>>& occupations,
        const std::vector<Vec3>& k_points,
        const std::vector<double>& k_weights,
        int spin_factor);

    static std::vector<Vec3> compute_total_forces(
        const std::vector<Vec3>& ewald_forces,
        const std::vector<Vec3>& local_forces,
        const std::vector<Vec3>& nonlocal_forces);
};
```

---

## Solver Module (`src/solver/`)

### SCFSolver (`solver/scf.hpp`)

```cpp
struct SCFResult {
    bool   converged;
    int    scf_steps;
    double total_energy_ry, total_energy_ev, fermi_energy_ev;
    double kinetic_energy, hartree_energy, xc_energy;
    double local_pp_energy, nonlocal_pp_energy, ewald_energy;
    std::vector<Vec3> forces, ewald_forces, local_forces, nonlocal_forces;
    std::vector<std::vector<double>> eigenvalues;  // [k][band] in Ry
    std::vector<complex_t> converged_veff_r;
    std::map<std::string, double> timing;
};

class SCFSolver {
public:
    SCFSolver(const Crystal& crystal,
              const CalculationParams& calc_params,
              const ConvergenceParams& conv_params,
              const std::map<std::string, PseudoPotential>& pseudopotentials);
    SCFResult solve();
};
```

**Example:**
```cpp
Crystal crystal = ...;
auto pps = ...;
CalculationParams calc;
calc.ecutwfc = 30.0;
calc.kpoints.grid = {4, 4, 4};
ConvergenceParams conv;
conv.energy_threshold = 1e-8;
SCFSolver solver(crystal, calc, conv, pps);
SCFResult result = solver.solve();
if (result.converged) {
    std::cout << "E = " << result.total_energy_ev << " eV\n";
}
```

### DavidsonSolver (`solver/davidson.hpp`)

```cpp
struct EigenResult {
    std::vector<double> eigenvalues;   // Sorted ascending (Ry)
    std::vector<CVec>   eigenvectors;  // Corresponding wavefunctions
    int  iterations;
    bool converged;
    double max_residual;
};

class DavidsonSolver {
public:
    struct Params {
        int    max_iterations  = 100;
        double tolerance       = 1e-6;
        int    subspace_factor = 3;
        int    max_subspace    = 0;     // 0 = auto
    };

    DavidsonSolver();
    explicit DavidsonSolver(Params params);

    EigenResult solve(
        const std::function<CVec(const CVec&)>& h_apply,
        const std::vector<double>& preconditioner,
        int num_bands, int num_pw,
        const std::vector<CVec>& initial_guess = {});
};
```

### FermiSolver (`solver/fermi.hpp`)

```cpp
struct FermiResult {
    double fermi_energy;                          // Ry
    std::vector<std::vector<double>> occupations;  // [k][band]
    double total_electrons_found;
    bool   converged;
};

class FermiSolver {
public:
    static FermiResult find_fermi_level(
        const std::vector<std::vector<double>>& eigenvalues,
        const std::vector<double>& weights,
        double target_electrons,
        SmearingType smearing, double degauss,
        int spin_factor = 2);
};
```

### Mixing (`solver/mixing.hpp`)

```cpp
class LinearMixer {
public:
    explicit LinearMixer(double alpha = 0.3);
    RVec mix(const RVec& n_in, const RVec& n_out);
};

class PulayMixer {
public:
    explicit PulayMixer(int max_history = 8, double alpha = 0.3);
    RVec mix(const RVec& n_in, const RVec& n_out);
    void reset();
    int  history_size() const;
};

class KerkerPreconditioner {
public:
    explicit KerkerPreconditioner(double q0 = 1.5);
    CVec apply(const CVec& residual_g,
               const std::vector<double>& g_norm2) const;
};
```

---

## Hamiltonian Module (`src/hamiltonian/`)

### Hamiltonian (`hamiltonian/hamiltonian.hpp`)

```cpp
class Hamiltonian {
public:
    Hamiltonian(const Crystal& crystal, const PlaneWaveBasis& basis,
                FFTGrid& fft_grid, const NonlocalPP& nonlocal_pp);

    void update_veff(const std::vector<complex_t>& veff_r);
    CVec apply(const CVec& psi_g, const Vec3& k_frac) const;
    std::function<CVec(const CVec&)> get_apply_function(const Vec3& k_frac) const;
    std::vector<double> kinetic_diagonal(const Vec3& k_frac) const;
};
```

**H|ψ⟩ computation:**
```
H|ψ⟩ = T|ψ⟩ + FFT⁻¹[V_eff · FFT[ψ]] + V_NL|ψ⟩

T|ψ⟩_G = |k+G|²/2 · ψ_G        (kinetic, pointwise)
V_eff·ψ in real space             (local, via FFT)
V_NL|ψ⟩ = Σ D_l |β⟩⟨β|ψ⟩       (nonlocal, via GEMM)
```

---

## PostProcessing Module (`src/postprocessing/`)

### BandStructureCalculator (`postprocessing/band_structure.hpp`)

```cpp
struct BandData {
    std::vector<Vec3>   kpoints;       // Fractional coords
    std::vector<double> distances;     // Cumulative k-path distance
    std::vector<std::vector<double>> eigenvalues;  // [nk][nbands] Ry
    std::vector<std::pair<double,std::string>> tick_positions;
};

class BandStructureCalculator {
public:
    static BandData generate_kpath(const Crystal& crystal,
                                   const KPathSpec& path_spec,
                                   int npoints_per_segment = 50);
    static void compute_bands(BandData& kpath, ...);
    static void write_bands_gnuplot(const std::string& filename,
                                    const BandData& band_data);
    static KPathSpec default_path_fcc();
    static KPathSpec default_path_bcc();
    static KPathSpec default_path_sc();
    static KPathSpec default_path_hcp();
};
```

### DOSCalculator (`postprocessing/dos.hpp`)

```cpp
struct DOSData {
    std::vector<double> energies;        // eV
    std::vector<double> dos_values;      // states/eV
    std::vector<double> integrated_dos;
};

class DOSCalculator {
public:
    static DOSData compute_dos(
        const std::vector<std::vector<double>>& eigenvalues,
        const std::vector<double>& weights,
        SmearingType smearing,
        double degauss = 0.05, double energy_min = -20.0,
        double energy_max = 20.0, int num_points = 2001,
        int spin_factor = 2);
    static void write_dos(const std::string& filename, const DOSData& data);
};
```

---

## Utils Module (`src/utils/`)

### Timer (`utils/timer.hpp`)

```cpp
// Scoped timer macro — records wall time in the global registry
#define KRONOS_TIMER(name)  kronos::ScopedTimer _timer_##__LINE__(name)

class ScopedTimer {
public:
    explicit ScopedTimer(const std::string& name);
    ~ScopedTimer();  // Records elapsed time
};

// Global timing registry
std::map<std::string, double> get_timing_results();
void reset_timing();
```

### Logger (`utils/logger.hpp`)

Structured JSON logging to stderr:

```cpp
namespace kronos::logger {
    void info(const std::string& event, ...);
    void warn(const std::string& event, ...);
    void error(const std::string& event, ...);
}
```

Output format:
```json
{"timestamp": "2024-01-15T10:30:00Z", "level": "info", "event": "scf_step", "step": 5, "energy": -15.847, "wall_s": 0.234}
```
