#pragma once
#include <array>
#include <complex>
#include <vector>
#include <string>
#include <map>

namespace kronos {

using real_t = double;
using complex_t = std::complex<double>;
using Vec3 = std::array<double, 3>;
using Mat3 = std::array<std::array<double, 3>, 3>;
using CVec = std::vector<complex_t>;  // complex wavefunction vector
using RVec = std::vector<double>;     // real-space grid vector

// Calculation type
enum class CalculationType { SCF, Relax, VCRelax, Bands, DOS };

// Smearing method for metals
enum class SmearingType { None, Gaussian, MarzariVanderbilt, FermiDirac };

// Eigensolver choice
enum class EigensolverType { Davidson, LOBPCG };

// K-point grid specification
struct KPointGrid {
    std::array<int, 3> grid{1, 1, 1};
    std::array<int, 3> shift{0, 0, 0};
};

// Atom in the crystal
struct Atom {
    std::string symbol;
    int atomic_number{0};
    Vec3 position{};  // fractional coordinates in [0,1)
};

// Calculation parameters
struct CalculationParams {
    CalculationType type{CalculationType::SCF};
    double ecutwfc{30.0};    // plane-wave cutoff in Ry
    double ecutrho{0.0};     // density cutoff in Ry (0 = auto)
    KPointGrid kpoints{};
    std::string xc_functional{"LDA_PZ"};
    SmearingType smearing{SmearingType::None};
    double degauss{0.01};    // smearing width in Ry
    bool spin_polarized{false};
    int nspin{1};            // 1 = unpolarized, 2 = spin-polarized (LSDA)
    std::map<std::string, double> starting_magnetization; // element -> initial mag [-1,1]
    EigensolverType eigensolver{EigensolverType::Davidson};

    // VC-relax parameters
    double press_target{0.0};  // Target pressure in GPa (0 = zero pressure)
    double cell_factor{2.0};   // Factor for cell optimization step size

    // Checkpoint/restart parameters
    int checkpoint_every{0};                         // 0 = disabled
    std::string checkpoint_file{"kronos_checkpoint.bin"};  // checkpoint filename
    bool restart_from_checkpoint{false};             // restart from existing checkpoint
    std::string input_hash;                          // hash of YAML input for verification
};

// Convergence criteria
struct ConvergenceParams {
    double energy_threshold{1e-8};    // Ry
    double density_threshold{1e-9};
    int max_scf_steps{100};
    double force_threshold{1e-3};     // Ry/bohr for geometry opt
    double stress_threshold{0.5};     // kbar for vc-relax convergence
};

// Hardware configuration
struct HardwareParams {
    bool use_gpu{false};
    std::string gpu_backend{"none"};  // "cuda", "hip", "none"
    int mpi_tasks{1};
};

// Complete input specification
struct InputData {
    // Crystal structure fields set separately
    CalculationParams calculation;
    ConvergenceParams convergence;
    HardwareParams hardware;
    std::map<std::string, std::string> pseudopotentials;  // symbol -> filepath
};

} // namespace kronos
