#include "solver/scf.hpp"
#include "basis/kpoints.hpp"
#include "hamiltonian/hamiltonian.hpp"
#include "potential/hartree.hpp"
#include "potential/xc.hpp"
#include "potential/gradient.hpp"
#include "potential/local_pp.hpp"
#include "potential/nonlocal_pp.hpp"
#include "potential/ewald.hpp"
#include "potential/forces.hpp"
#include "potential/stress.hpp"
#include "potential/paw.hpp"
#include "potential/exact_exchange.hpp"
#include "io/checkpoint.hpp"
#include "solver/davidson.hpp"
#include "solver/mixing.hpp"
#include "solver/fermi.hpp"
#include "core/constants.hpp"
#include "utils/timer.hpp"
#include "utils/logger.hpp"
#include "utils/radial_integral.hpp"
#include "utils/mpi_wrapper.hpp"
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <map>
#include <numeric>

#ifdef KRONOS_HAS_SPGLIB
extern "C" {
#include <spglib.h>
}
#endif

namespace kronos {

// =========================================================================
// Force symmetrization using spglib symmetry operations
// =========================================================================
//
// Forces computed from IBZ k-points don't have full crystal symmetry.
// Symmetrize by averaging over all point group operations:
//   F_sym(i) = (1/N_ops) * sum_{(R,t)} R_cart * F_raw(j)
// where atom j maps to atom i under operation (R, t).
// =========================================================================

#ifdef KRONOS_HAS_SPGLIB
static std::vector<Vec3> symmetrize_forces(
    const Crystal& crystal,
    const std::vector<Vec3>& forces_raw)
{
    const size_t natoms = crystal.num_atoms();
    if (natoms == 0) return forces_raw;

    // Prepare crystal structure for spglib
    const auto& lat = crystal.lattice();
    double lattice[3][3];
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            lattice[i][j] = lat[i][j];

    std::vector<std::array<double, 3>> positions(natoms);
    std::vector<int> types(natoms);
    std::map<std::string, int> symbol_to_type;
    int next_type = 0;
    for (size_t ia = 0; ia < natoms; ++ia) {
        const auto& atom = crystal.atom(ia);
        auto it = symbol_to_type.find(atom.symbol);
        if (it == symbol_to_type.end()) {
            symbol_to_type[atom.symbol] = next_type;
            types[ia] = next_type++;
        } else {
            types[ia] = it->second;
        }
        positions[ia] = {atom.position[0], atom.position[1], atom.position[2]};
    }

    // Get symmetry operations
    const int max_ops = 192;  // max for cubic
    int rotations[192][3][3];
    double translations[192][3];
    int num_ops = spg_get_symmetry(
        rotations, translations,
        max_ops,
        lattice,
        reinterpret_cast<double(*)[3]>(positions.data()),
        types.data(),
        static_cast<int>(natoms),
        1e-5);

    if (num_ops <= 0) {
        // spglib failed — return unsymmetrized forces
        return forces_raw;
    }

    // For each operation (R, t), find atom mapping:
    // R * pos[j] + t = pos[i] (mod lattice)
    // We precompute the mapping for all operations.
    std::vector<std::vector<int>> atom_map(num_ops, std::vector<int>(natoms, -1));
    for (int iop = 0; iop < num_ops; ++iop) {
        for (size_t j = 0; j < natoms; ++j) {
            // Compute R * pos[j] + t
            double rp[3];
            for (int d = 0; d < 3; ++d) {
                rp[d] = translations[iop][d];
                for (int d2 = 0; d2 < 3; ++d2) {
                    rp[d] += rotations[iop][d][d2] * positions[j][d2];
                }
                // Wrap to [0, 1)
                rp[d] = rp[d] - std::floor(rp[d]);
            }

            // Find matching atom i
            for (size_t i = 0; i < natoms; ++i) {
                if (types[i] != types[j]) continue;
                bool match = true;
                for (int d = 0; d < 3; ++d) {
                    double diff = rp[d] - positions[i][d];
                    diff -= std::round(diff);
                    if (std::abs(diff) > 1e-4) { match = false; break; }
                }
                if (match) {
                    atom_map[iop][j] = static_cast<int>(i);
                    break;
                }
            }
        }
    }

    // Symmetrize forces:
    // F_sym(i) = (1/N_ops) * sum_{op} R_cart * F_raw(j)
    // where j is the atom that maps TO i under this operation
    // i.e., R*pos[j] + t = pos[i], so atom_map[op][j] = i
    // We need the inverse: for each i, which j maps to i?
    std::vector<Vec3> forces_sym(natoms, {0.0, 0.0, 0.0});

    for (int iop = 0; iop < num_ops; ++iop) {
        for (size_t j = 0; j < natoms; ++j) {
            int i = atom_map[iop][j];
            if (i < 0) continue;

            // Rotate force on atom j using R_cart:
            // 1. Convert F_cart to fractional direction
            Vec3 f_frac = crystal.cart_to_frac(forces_raw[j]);
            // 2. Apply R_frac in fractional space
            Vec3 f_frac_rot{};
            for (int d = 0; d < 3; ++d) {
                for (int d2 = 0; d2 < 3; ++d2) {
                    f_frac_rot[d] += rotations[iop][d][d2] * f_frac[d2];
                }
            }
            // 3. Convert back to Cartesian
            Vec3 f_cart_rot = crystal.frac_to_cart(f_frac_rot);

            // Accumulate into symmetrized force for atom i
            for (int d = 0; d < 3; ++d) {
                forces_sym[i][d] += f_cart_rot[d];
            }
        }
    }

    // Divide by number of operations
    for (size_t i = 0; i < natoms; ++i) {
        for (int d = 0; d < 3; ++d) {
            forces_sym[i][d] /= static_cast<double>(num_ops);
        }
    }

    return forces_sym;
}

// =========================================================================
// Density symmetrization in G-space using spglib symmetry operations
// =========================================================================
//
// When using IBZ (irreducible Brillouin zone) k-points, the electron
// density computed from wavefunctions lacks the full crystal symmetry.
// QE symmetrizes n(G) at every SCF step; this function does the same.
//
// Algorithm (QE convention):
//   n_sym(G) = (1/N_ops) * sum_{S} exp(-i 2pi t_S . G) * n(S^{-1} G)
//
// where {S, t_S} are the space group operations from spglib.
// S is a 3x3 integer matrix (rotation in fractional coordinates),
// t_S is the fractional translation, and S^{-1} G means the inverse
// rotation applied to the Miller index vector G.
//
// For integer matrices with det(S) = +/-1, the inverse is computed
// via the adjugate matrix: S^{-1} = det(S) * adj(S).
// =========================================================================

static void symmetrize_density_g(
    const Crystal& crystal,
    std::vector<complex_t>& density_g,
    const std::array<int, 3>& grid_dims)
{
    const int n0 = grid_dims[0];
    const int n1 = grid_dims[1];
    const int n2 = grid_dims[2];
    const int num_grid = n0 * n1 * n2;

    if (num_grid == 0) return;

    // Prepare crystal structure for spglib
    const size_t natoms = crystal.num_atoms();
    const auto& lat = crystal.lattice();
    double lattice[3][3];
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            lattice[i][j] = lat[i][j];

    std::vector<std::array<double, 3>> positions(natoms);
    std::vector<int> types(natoms);
    std::map<std::string, int> symbol_to_type;
    int next_type = 0;
    for (size_t ia = 0; ia < natoms; ++ia) {
        const auto& atom = crystal.atom(ia);
        auto it = symbol_to_type.find(atom.symbol);
        if (it == symbol_to_type.end()) {
            symbol_to_type[atom.symbol] = next_type;
            types[ia] = next_type++;
        } else {
            types[ia] = it->second;
        }
        positions[ia] = {atom.position[0], atom.position[1], atom.position[2]};
    }

    // Get symmetry operations
    const int max_ops = 192;
    int rotations[192][3][3];
    double translations[192][3];
    int num_ops = spg_get_symmetry(
        rotations, translations,
        max_ops,
        lattice,
        reinterpret_cast<double(*)[3]>(positions.data()),
        types.data(),
        static_cast<int>(natoms),
        1e-5);

    if (num_ops <= 1) {
        // Only identity operation (or spglib failed) — nothing to symmetrize
        return;
    }

    // Save G=0 component: it must remain unchanged (total electron count)
    complex_t n_g0 = density_g[0];


    // Symmetrize density in G-space.
    //
    // The space group operation {S, t} in fractional coordinates transforms
    // direct-space points as r' = S r + t.  The symmetrization projector
    // for Fourier coefficients is:
    //
    //   n_sym(G) = (1/N_ops) sum_{S,t} exp(-i 2pi G.t) * n((S^{-1})^T G)
    //
    // The reciprocal-space transformation is G' = (S^{-1})^T G, NOT S^T G.
    // These are only equivalent when S is orthogonal (S^{-1} = S^T), which
    // is NOT the case for fractional-coordinate rotations on non-orthogonal
    // lattices like FCC.
    //
    // For each operation, we precompute the inverse rotation matrix.
    // Since |det(S)| = 1, the inverse is: S^{-1} = det(S) * adj(S).
    //
    std::vector<complex_t> density_sym(num_grid, complex_t{0.0, 0.0});

    for (int iop = 0; iop < num_ops; ++iop) {
        const auto& S = rotations[iop];
        const auto& t = translations[iop];

        // Compute S^{-1} via adjugate: S^{-1} = det(S) * adj(S)
        int det = S[0][0] * (S[1][1]*S[2][2] - S[1][2]*S[2][1])
                - S[0][1] * (S[1][0]*S[2][2] - S[1][2]*S[2][0])
                + S[0][2] * (S[1][0]*S[2][1] - S[1][1]*S[2][0]);
        // adjugate (cofactor matrix transposed)
        int Sinv[3][3];
        Sinv[0][0] = det * (S[1][1]*S[2][2] - S[1][2]*S[2][1]);
        Sinv[0][1] = det * (S[0][2]*S[2][1] - S[0][1]*S[2][2]);
        Sinv[0][2] = det * (S[0][1]*S[1][2] - S[0][2]*S[1][1]);
        Sinv[1][0] = det * (S[1][2]*S[2][0] - S[1][0]*S[2][2]);
        Sinv[1][1] = det * (S[0][0]*S[2][2] - S[0][2]*S[2][0]);
        Sinv[1][2] = det * (S[0][2]*S[1][0] - S[0][0]*S[1][2]);
        Sinv[2][0] = det * (S[1][0]*S[2][1] - S[1][1]*S[2][0]);
        Sinv[2][1] = det * (S[0][1]*S[2][0] - S[0][0]*S[2][1]);
        Sinv[2][2] = det * (S[0][0]*S[1][1] - S[0][1]*S[1][0]);

        for (int idx = 0; idx < num_grid; ++idx) {
            // Decode linear index to Miller indices (h, k, l)
            int hi = idx / (n1 * n2);
            int ki = (idx % (n1 * n2)) / n2;
            int li = idx % n2;
            int h = (hi <= n0 / 2) ? hi : hi - n0;
            int k = (ki <= n1 / 2) ? ki : ki - n1;
            int l = (li <= n2 / 2) ? li : li - n2;

            // Rotated indices: G' = (S^{-1})^T * G
            // (S^{-1})^T[i][j] = Sinv[j][i]
            int hp = Sinv[0][0] * h + Sinv[1][0] * k + Sinv[2][0] * l;
            int kp = Sinv[0][1] * h + Sinv[1][1] * k + Sinv[2][1] * l;
            int lp = Sinv[0][2] * h + Sinv[1][2] * k + Sinv[2][2] * l;

            // Wrap rotated indices to FFT grid range [0, n)
            int hi_p = ((hp % n0) + n0) % n0;
            int ki_p = ((kp % n1) + n1) % n1;
            int li_p = ((lp % n2) + n2) % n2;

            int idx_rot = hi_p * (n1 * n2) + ki_p * n2 + li_p;

            // Phase factor: exp(-i 2pi t . G)
            double phase_arg = -2.0 * constants::pi * (t[0] * h + t[1] * k + t[2] * l);
            complex_t phase{std::cos(phase_arg), std::sin(phase_arg)};

            density_sym[idx] += density_g[idx_rot] * phase;
        }
    }

    // Divide by number of operations
    double inv_nops = 1.0 / static_cast<double>(num_ops);
    for (int idx = 0; idx < num_grid; ++idx) {
        density_g[idx] = density_sym[idx] * inv_nops;
    }

    // Restore G=0 exactly (preserve total electron count)
    density_g[0] = n_g0;

}
#endif

SCFSolver::SCFSolver(const Crystal& crystal,
                     const CalculationParams& calc_params,
                     const ConvergenceParams& conv_params,
                     const std::map<std::string, PseudoPotential>& pseudopotentials)
    : crystal_(crystal)
    , calc_params_(calc_params)
    , conv_params_(conv_params)
    , pseudopotentials_(pseudopotentials)
{
}

int SCFSolver::compute_num_bands() const {
    // Count total valence electrons from pseudopotentials
    double total_valence = 0.0;
    for (const auto& atom : crystal_.atoms()) {
        auto it = pseudopotentials_.find(atom.symbol);
        if (it != pseudopotentials_.end()) {
            total_valence += it->second.z_valence;
        }
    }
    // Number of occupied bands + some empty bands for convergence
    int num_occupied = static_cast<int>(std::ceil(total_valence / 2.0));
    return std::max(num_occupied + 4, 8);
}

RVec SCFSolver::initial_density(const PlaneWaveBasis& basis,
                                FFTGrid& fft_grid) const {
    // Superposition of atomic charge densities.
    //
    // For each G-vector:
    //   n(G) = sum_atoms  rho_atom_species(|G|) * exp(-i G . tau_atom)
    //
    // where rho_atom_species(q) is the radial Fourier transform of the
    // atomic charge density from the UPF file:
    //   rho_atom(q) = (4*pi/Omega) * integral r^2 rho_atomic(r) sinc(qr) rab dr

    const double volume = crystal_.volume();
    const auto& gvecs = basis.gvectors();
    const size_t npw = gvecs.size();
    const int num_grid = fft_grid.total_points();

    // Compute total valence electrons for normalization
    double total_valence = 0.0;
    for (const auto& atom : crystal_.atoms()) {
        auto it = pseudopotentials_.find(atom.symbol);
        if (it != pseudopotentials_.end()) {
            total_valence += it->second.z_valence;
        }
    }

    // If no pseudopotentials have nonzero rho_atomic data, fall back to uniform density
    bool have_rho_atomic = false;
    for (const auto& [symbol, pp] : pseudopotentials_) {
        if (!pp.rho_atomic.empty()) {
            // Check if rho_atomic is actually nonzero (some PPs have all-zero data)
            double rho_sum = 0.0;
            for (double v : pp.rho_atomic) rho_sum += std::abs(v);
            if (rho_sum > 1e-20) {
                have_rho_atomic = true;
                break;
            }
        }
    }

    if (!have_rho_atomic) {
        double uniform_density = total_valence / volume;
        return RVec(num_grid, uniform_density);
    }

    // Group atoms by species: species -> list of Cartesian positions (bohr)
    std::map<std::string, std::vector<Vec3>> species_positions;
    for (size_t ia = 0; ia < crystal_.num_atoms(); ++ia) {
        const auto& atom = crystal_.atom(ia);
        Vec3 cart = crystal_.frac_to_cart(atom.position);
        species_positions[atom.symbol].push_back(cart);
    }

    // Build n(G) in the plane-wave basis
    CVec density_pw(npw, complex_t{0.0, 0.0});

    for (size_t ig = 0; ig < npw; ++ig) {
        const Vec3& g_cart = gvecs[ig].cart;
        const double g_mag = std::sqrt(gvecs[ig].norm2);

        complex_t n_g{0.0, 0.0};

        for (const auto& [symbol, positions] : species_positions) {
            auto pp_it = pseudopotentials_.find(symbol);
            if (pp_it == pseudopotentials_.end()) continue;

            const auto& pp = pp_it->second;
            if (pp.rho_atomic.empty()) continue;

            const auto& r   = pp.mesh.r;
            const auto& rab = pp.mesh.rab;
            const int npts = pp.mesh.npoints;

            // Radial Fourier transform of rho_atomic at |G|.
            // UPF convention: rho_atomic stores 4*pi*r^2*rho(r), so
            // n(q) = 4*pi * integral r^2 rho(r) sinc(qr) dr
            //      = integral rho_atomic(r) sinc(qr) dr
            // n_phys(G) = n(q) / Omega
            std::vector<double> rho_integrand(npts);

            if (g_mag < 1.0e-12) {
                // G = 0: sinc(0) = 1
                for (int i = 0; i < npts; ++i) {
                    rho_integrand[i] = pp.rho_atomic[i];
                }
            } else {
                for (int i = 0; i < npts; ++i) {
                    const double ri = r[i];
                    if (ri < 1.0e-30) { rho_integrand[i] = 0.0; continue; }
                    const double qr = g_mag * ri;
                    const double sinc_qr = std::sin(qr) / qr;
                    rho_integrand[i] = pp.rho_atomic[i] * sinc_qr;
                }
            }
            double integral = simpson_radial(rho_integrand, rab, npts);

            double rho_atom_g = integral / volume;

            // Structure factor: S(G) = sum_j exp(-i G . tau_j)
            for (const auto& tau : positions) {
                const double gdottau = g_cart[0] * tau[0]
                                     + g_cart[1] * tau[1]
                                     + g_cart[2] * tau[2];
                n_g += rho_atom_g * complex_t{std::cos(gdottau), -std::sin(gdottau)};
            }
        }

        density_pw[ig] = n_g;
    }

    // Scatter n(G) onto full FFT grid and inverse FFT to get n(r)
    std::vector<complex_t> density_g_grid(num_grid, complex_t{0.0, 0.0});
    fft_grid.scatter_to_grid(basis, density_pw, density_g_grid);

    std::vector<complex_t> density_c(num_grid);
    fft_grid.inverse(density_g_grid, density_c);

    // Extract real part and clamp negatives
    RVec density_r(num_grid);
    for (int i = 0; i < num_grid; ++i) {
        density_r[i] = std::max(0.0, std::real(density_c[i]));
    }

    // Normalize so that integral n(r) dr = total_electrons
    // integral = sum_i n(r_i) * (Omega / N_grid)
    double dn_sum = 0.0;
    for (int i = 0; i < num_grid; ++i) {
        dn_sum += density_r[i];
    }
    if (dn_sum > 1.0e-15) {
        double scale = total_valence / (dn_sum * volume / num_grid);
        for (int i = 0; i < num_grid; ++i) {
            density_r[i] *= scale;
        }
    }

    return density_r;
}

double SCFSolver::compute_total_energy(double kinetic, double hartree, double xc,
                                       double local_pp, double nonlocal_pp,
                                       double ewald) const {
    return kinetic + hartree + xc + local_pp + nonlocal_pp + ewald;
}

double SCFSolver::compute_band_energy(
    const std::vector<std::vector<double>>& eigenvalues,
    const std::vector<std::vector<double>>& occupations,
    const std::vector<double>& kweights) const {
    double e_band = 0.0;
    for (size_t ik = 0; ik < eigenvalues.size(); ++ik) {
        for (size_t n = 0; n < eigenvalues[ik].size(); ++n) {
            e_band += kweights[ik] * occupations[ik][n] * eigenvalues[ik][n];
        }
    }
    return e_band;
}

void SCFSolver::print_scf_step(int step, double energy, double de, double dn,
                                double wall_time) const {
    if (step == 1) {
        std::printf("SCF step %2d: E = %12.6f Ry  |dE| = ---        |dn| = %.2e  t = %.1fs\n",
                    step, energy, dn, wall_time);
    } else {
        std::printf("SCF step %2d: E = %12.6f Ry  |dE| = %.2e  |dn| = %.2e  t = %.1fs\n",
                    step, energy, de, dn, wall_time);
    }
}

SCFResult SCFSolver::solve() {
    KRONOS_TIMER("total_scf");

    auto& logger = Logger::instance();

    // MPI rank/size for k-point parallelization
    const int mpi_rank = mpi::rank();
    const int mpi_size = mpi::size();
    const bool is_root = (mpi_rank == 0);

    // 1. Generate k-points first (needed to determine basis expansion)
    auto kpoint_data = KPointGenerator::generate_monkhorst_pack(
        calc_params_.kpoints, crystal_);
    std::vector<Vec3> kpoints = kpoint_data.kpoints;
    std::vector<double> kweights = kpoint_data.weights;
    const int nk_total = static_cast<int>(kpoints.size());

    // Distribute k-points across MPI ranks (round-robin).
    // my_kpoint_indices[i] is a global k-point index handled by this rank.
    std::vector<int> my_kpoint_indices;
    for (int ik = 0; ik < nk_total; ++ik) {
        if (ik % mpi_size == mpi_rank) {
            my_kpoint_indices.push_back(ik);
        }
    }
    const int nk_local = static_cast<int>(my_kpoint_indices.size());

    // Compute k_max = max |k_cart| over all k-points for basis expansion.
    // The shared basis must include all G where |k+G|^2 <= ecutwfc for any k.
    const Mat3& recip_lat = crystal_.reciprocal_lattice();
    double k_max = 0.0;
    for (const auto& kf : kpoints) {
        double kc0 = kf[0]*recip_lat[0][0] + kf[1]*recip_lat[1][0] + kf[2]*recip_lat[2][0];
        double kc1 = kf[0]*recip_lat[0][1] + kf[1]*recip_lat[1][1] + kf[2]*recip_lat[2][1];
        double kc2 = kf[0]*recip_lat[0][2] + kf[1]*recip_lat[1][2] + kf[2]*recip_lat[2][2];
        double knorm = std::sqrt(kc0*kc0 + kc1*kc1 + kc2*kc2);
        if (knorm > k_max) k_max = knorm;
    }

    // 2. Set up basis and FFT grid (expanded to cover all k-points)
    PlaneWaveBasis basis(crystal_, calc_params_.ecutwfc, k_max);
    double ecutrho = calc_params_.ecutrho > 0 ? calc_params_.ecutrho : 4.0 * calc_params_.ecutwfc;
    FFTGrid fft_grid(basis, ecutrho);

    auto grid_dims = fft_grid.dims();
    logger.info("basis", "Plane-wave basis constructed",
        {{"num_pw", std::to_string(basis.num_pw())},
         {"k_max", std::to_string(k_max)},
         {"fft_grid", std::to_string(grid_dims[0]) + "x" +
                      std::to_string(grid_dims[1]) + "x" +
                      std::to_string(grid_dims[2])}});

    // 3. Set up potentials
    HartreeSolver hartree(basis);
    XCEvaluator xc(calc_params_.xc_functional);
    LocalPPEvaluator local_pp(crystal_, basis, pseudopotentials_);
    NonlocalPP nonlocal_pp(crystal_, basis, pseudopotentials_);
    Hamiltonian ham(crystal_, basis, fft_grid, nonlocal_pp);

    // PAW calculator (active only when PAW PPs are present)
    PAWCalculator paw_calc(crystal_, basis, fft_grid, pseudopotentials_);
    bool use_paw = paw_calc.has_paw();
    if (use_paw && is_root) {
        std::printf("  PAW pseudopotentials detected\n");
        // Auto-adjust ecutrho for PAW if not explicitly set
        if (calc_params_.ecutrho <= 0 || calc_params_.ecutrho < 12.0 * calc_params_.ecutwfc) {
            ecutrho = 12.0 * calc_params_.ecutwfc;
            std::printf("  PAW: ecutrho auto-set to 12*ecutwfc = %.1f Ry\n", ecutrho);
        }
    }

    // Save base D_ij for PAW reset/restore cycle
    if (use_paw) {
        nonlocal_pp.save_base_dij();
    }

    // Build PAW atom → NonlocalPP atom index mapping
    std::vector<size_t> paw_atom_to_nlpp_idx;
    if (use_paw) {
        // PAWCalculator and NonlocalPP both iterate over crystal atoms in order,
        // but only include atoms that have PAW/nonlocal PPs respectively.
        // Build mapping by matching crystal atom indices.
        for (size_t ia_paw = 0; ia_paw < paw_calc.rho_ij().size() || ia_paw == 0; ++ia_paw) {
            // We'll rebuild this after we know the sizes
            (void)ia_paw;
        }
        paw_atom_to_nlpp_idx.clear();
        // Iterate PAW atoms and find matching NLPP atom by crystal index
        // PAW atoms are indexed through the PAWCalculator internal order.
        // For now, assume they're in the same crystal order (both iterate crystal_.atoms()).
        // Build a map from crystal_atom_index -> nlpp_atom_index
        std::map<int, size_t> crystal_to_nlpp;
        for (size_t ia = 0; ia < nonlocal_pp.num_atoms(); ++ia) {
            crystal_to_nlpp[nonlocal_pp.crystal_atom_index(ia)] = ia;
        }
        // PAW atoms correspond to crystal atoms with PAW PPs.
        // We iterate in the same order PAWCalculator stores them.
        for (size_t ia = 0; ia < crystal_.num_atoms(); ++ia) {
            const auto& atom = crystal_.atom(ia);
            auto pp_it = pseudopotentials_.find(atom.symbol);
            if (pp_it == pseudopotentials_.end()) continue;
            if (!pp_it->second.is_paw || !pp_it->second.paw.has_value()) continue;
            auto nlpp_it = crystal_to_nlpp.find(static_cast<int>(ia));
            if (nlpp_it != crystal_to_nlpp.end()) {
                paw_atom_to_nlpp_idx.push_back(nlpp_it->second);
            }
        }
    }

    // Exact exchange for hybrid functionals
    bool use_hybrid = xc.is_hybrid();
    std::unique_ptr<ExactExchange> exx;
    if (use_hybrid) {
        exx = std::make_unique<ExactExchange>(
            crystal_, basis, fft_grid,
            xc.hybrid_type(),
            calc_params_.exx_fraction,
            calc_params_.screening_parameter);
        // Scale semi-local exchange by (1 - α) for hybrid functionals
        xc.set_exchange_scale(1.0 - calc_params_.exx_fraction);
        if (is_root) {
            std::printf("  Hybrid functional: %s (alpha=%.2f)\n",
                        calc_params_.xc_functional.c_str(), calc_params_.exx_fraction);
        }
    }

    // 4. Set up solver components
    PulayMixer mixer(8, 0.2);
    DavidsonSolver eigensolver;

    // Kerker preconditioner for metals (activated when smearing is used)
    bool use_kerker = (calc_params_.smearing != SmearingType::None);
    KerkerPreconditioner kerker(1.5);
    // Precompute |G|^2 for Kerker
    std::vector<double> g_norm2_pw;
    if (use_kerker) {
        const auto& gvecs = basis.gvectors();
        g_norm2_pw.resize(gvecs.size());
        for (size_t ig = 0; ig < gvecs.size(); ++ig) {
            g_norm2_pw[ig] = gvecs[ig].norm2;
        }
    }

    int num_bands = compute_num_bands();
    int num_pw = static_cast<int>(basis.num_pw());
    double volume = crystal_.volume();
    int num_grid = fft_grid.total_points();
    int spin_factor = calc_params_.spin_polarized ? 1 : 2;

    // ------------------------------------------------------------------
    // Pre-compute G² and G_cart for every point on the full FFT grid.
    // V_H and V_loc must be evaluated on the full density grid (G² ≤
    // ecutrho) — not just the wavefunction PW basis — so that V_eff(r)
    // includes high-G components.  This matches QE's convention and is
    // essential for quantitative accuracy.
    // ------------------------------------------------------------------
    const int n0g = grid_dims[0], n1g = grid_dims[1], n2g = grid_dims[2];
    std::vector<double> grid_g2(num_grid);
    std::vector<Vec3> grid_gcart(num_grid);
    for (int idx = 0; idx < num_grid; ++idx) {
        int hi = idx / (n1g * n2g);
        int ki = (idx % (n1g * n2g)) / n2g;
        int li = idx % n2g;
        int h = (hi <= n0g / 2) ? hi : hi - n0g;
        int k = (ki <= n1g / 2) ? ki : ki - n1g;
        int l = (li <= n2g / 2) ? li : li - n2g;
        double gx = h * recip_lat[0][0] + k * recip_lat[1][0] + l * recip_lat[2][0];
        double gy = h * recip_lat[0][1] + k * recip_lat[1][1] + l * recip_lat[2][1];
        double gz = h * recip_lat[0][2] + k * recip_lat[1][2] + l * recip_lat[2][2];
        grid_g2[idx] = gx * gx + gy * gy + gz * gz;
        grid_gcart[idx] = {gx, gy, gz};
    }

    // Pre-compute V_loc on the full FFT grid (physics convention: /Ω).
    // Only include G² ≤ ecutrho to avoid aliased contributions.
    std::map<std::string, std::vector<Vec3>> species_positions_cart;
    for (size_t ia = 0; ia < crystal_.num_atoms(); ++ia) {
        const auto& atom = crystal_.atom(ia);
        Vec3 cart = crystal_.frac_to_cart(atom.position);
        species_positions_cart[atom.symbol].push_back(cart);
    }

    std::vector<complex_t> vloc_full_g(num_grid, {0.0, 0.0});
    for (int idx = 0; idx < num_grid; ++idx) {
        if (grid_g2[idx] > ecutrho + 1.0e-6) continue;
        double g_mag = std::sqrt(grid_g2[idx]);
        complex_t vtot{0.0, 0.0};
        for (const auto& [symbol, positions] : species_positions_cart) {
            auto pp_it = pseudopotentials_.find(symbol);
            if (pp_it == pseudopotentials_.end()) continue;
            double vq = LocalPPEvaluator::vloc_of_q(pp_it->second, g_mag, volume);
            complex_t sf = LocalPPEvaluator::structure_factor(positions, grid_gcart[idx]);
            vtot += vq * sf;
        }
        vloc_full_g[idx] = vtot;
    }

    // Compute target electron count
    double target_electrons = 0.0;
    for (const auto& atom : crystal_.atoms()) {
        auto it = pseudopotentials_.find(atom.symbol);
        if (it != pseudopotentials_.end()) {
            target_electrons += it->second.z_valence;
        }
    }

    logger.info("scf", "SCF solver initialized",
        {{"num_bands", std::to_string(num_bands)},
         {"num_pw", std::to_string(num_pw)},
         {"target_electrons", std::to_string(target_electrons)},
         {"num_kpoints", std::to_string(nk_total)},
         {"mpi_size", std::to_string(mpi_size)},
         {"nk_local", std::to_string(nk_local)}});

    // Log active PW count per k-point (G-vectors where |k+G|^2 <= ecutwfc)
    if (is_root) {
        for (int ik = 0; ik < nk_total; ++ik) {
            auto ke = basis.kinetic_energies(kpoints[ik]);
            int active = 0;
            for (size_t ig = 0; ig < ke.size(); ++ig) {
                if (ke[ig] <= calc_params_.ecutwfc + 1.0e-6) active++;
            }
            std::printf("  k-point %d: (%7.4f %7.4f %7.4f) weight=%.4f  npw=%d/%d  rank=%d\n",
                        ik, kpoints[ik][0], kpoints[ik][1], kpoints[ik][2],
                        kweights[ik], active, num_pw, ik % mpi_size);
        }
        if (mpi_size > 1) {
            std::printf("  MPI: %d processes, %d k-points/rank (approx)\n",
                        mpi_size, nk_local);
        }
    }

    // 5. Initialize density
    RVec density_r = initial_density(basis, fft_grid);

    // For spin-polarized (nspin=2): split into up/down densities
    const int nspin = calc_params_.nspin;
    RVec density_up_r, density_dn_r;
    if (nspin == 2) {
        density_up_r.resize(num_grid);
        density_dn_r.resize(num_grid);
        // Apply starting magnetization: n_up = n/2*(1+m), n_dn = n/2*(1-m)
        // Average magnetization from per-element starting_magnetization
        double avg_mag = 0.3;  // default starting magnetization
        if (!calc_params_.starting_magnetization.empty()) {
            double total_z = 0.0, weighted_mag = 0.0;
            for (const auto& atom : crystal_.atoms()) {
                auto it_pp = pseudopotentials_.find(atom.symbol);
                double zv = (it_pp != pseudopotentials_.end()) ? it_pp->second.z_valence : 0.0;
                auto it_mag = calc_params_.starting_magnetization.find(atom.symbol);
                double m = (it_mag != calc_params_.starting_magnetization.end()) ? it_mag->second : 0.0;
                total_z += zv;
                weighted_mag += zv * m;
            }
            if (total_z > 0.0) avg_mag = weighted_mag / total_z;
        }
        for (int i = 0; i < num_grid; ++i) {
            density_up_r[i] = density_r[i] * 0.5 * (1.0 + avg_mag);
            density_dn_r[i] = density_r[i] * 0.5 * (1.0 - avg_mag);
        }
        if (is_root) {
            std::printf("  Spin-polarized: nspin=2, starting magnetization=%.3f\n", avg_mag);
        }
    }

    // 6. SCF loop
    double prev_energy = 0.0;
    SCFResult result;

    // Retain converged data for post-SCF force/stress calculation
    std::vector<std::vector<CVec>> converged_wavefunctions;
    std::vector<std::vector<double>> converged_occupations;
    CVec converged_density_g;
    std::vector<complex_t> converged_density_g_full;
    std::vector<complex_t> converged_veff_r;
    double converged_exc_energy = 0.0;
    RVec converged_vxc_r;
    RVec converged_density_r;

    // LSDA mixing: mix total density and magnetization separately.
    // This is more stable than mixing n_up/n_dn independently.
    // Magnetization uses a simple linear mixer (no DIIS extrapolation)
    // to avoid losing spin polarization during early SCF steps.
    LinearMixer mixer_mag(0.2);

    // Outer loop for hybrid functionals: re-update ACE vectors after inner SCF converges
    const int outer_max = use_hybrid ? 8 : 1;
    double outer_energy_prev = 0.0;

    for (int outer = 0; outer < outer_max; ++outer) {
    if (use_hybrid && outer > 0) {
        // Collect occupied states from converged inner SCF and update ACE
        // For v0.8, assert serial (MPI + hybrid not supported)
        assert(mpi_size == 1 && "MPI + hybrid functionals not supported in v0.8");

        std::vector<Vec3> ace_kpts(nk_local);
        std::vector<double> ace_kwts(nk_local);
        for (int iloc = 0; iloc < nk_local; ++iloc) {
            int ik_global = my_kpoint_indices[iloc];
            ace_kpts[iloc] = kpoints[ik_global];
            ace_kwts[iloc] = kweights[ik_global];
        }
        exx->update_ace(converged_wavefunctions, converged_occupations,
                        ace_kpts, ace_kwts);

        // Reset mixer — potential landscape changed with new ACE vectors
        mixer.reset();
        if (is_root) {
            std::printf("  Hybrid outer SCF step %d: ACE vectors updated\n", outer);
        }
    }

    for (int step = 1; step <= conv_params_.max_scf_steps; ++step) {
        KRONOS_TIMER("scf_step");
        auto step_start = std::chrono::high_resolution_clock::now();

        // a. Compute total density in G-space via FFT
        //    For nspin=2: n_total = n_up + n_dn
        if (nspin == 2) {
            for (int i = 0; i < num_grid; ++i) {
                density_r[i] = density_up_r[i] + density_dn_r[i];
            }
        }

        std::vector<complex_t> density_c(num_grid);
        for (int i = 0; i < num_grid; ++i) {
            density_c[i] = complex_t{density_r[i], 0.0};
        }
        std::vector<complex_t> density_g_full(num_grid);
        fft_grid.forward(density_c, density_g_full);

        // NOTE: Density symmetrization is applied to the OUTPUT density
        // (after wavefunction-based density construction, before mixing),
        // not here at the input. See the symmetrization block before the
        // mixer call below.

        // Gather to PW coefficients
        CVec density_g(num_pw);
        fft_grid.gather_from_grid(basis, density_g_full, density_g);

        // PAW: add augmentation charge density to n(G) before Hartree/XC
        if (use_paw && step > 1) {
            paw_calc.add_augmentation_density(density_g_full, grid_gcart, grid_g2, ecutrho);
        }

        // b. Compute Hartree potential on the full FFT grid.
        //    V_H(G) = 8π n(G) / G²  (Rydberg units).
        constexpr double hartree_pf = 2.0 * constants::four_pi;  // 8π
        const double ng = static_cast<double>(num_grid);
        std::vector<complex_t> vh_full_g(num_grid, {0.0, 0.0});
        for (int i = 0; i < num_grid; ++i) {
            if (grid_g2[i] < 1.0e-12 || grid_g2[i] > ecutrho + 1.0e-6)
                continue;
            vh_full_g[i] = hartree_pf * density_g_full[i] / grid_g2[i];
        }

        // c. Compute XC potential on real-space grid
        XCResult xc_result;
        XCEvaluator::SpinXCResult spin_xc_result;
        double e_xc = 0.0;

        if (nspin == 2 && xc.is_gga()) {
            // Spin-polarized GGA: compute per-spin gradients and XC
            // 1. FFT spin densities to G-space
            std::vector<complex_t> dup_c(num_grid), ddn_c(num_grid);
            for (int i = 0; i < num_grid; ++i) {
                dup_c[i] = complex_t{density_up_r[i], 0.0};
                ddn_c[i] = complex_t{density_dn_r[i], 0.0};
            }
            std::vector<complex_t> dup_g_full(num_grid), ddn_g_full(num_grid);
            fft_grid.forward(dup_c, dup_g_full);
            fft_grid.forward(ddn_c, ddn_g_full);

            // 2. Gather to PW coefficients
            CVec dup_g(num_pw), ddn_g(num_pw);
            fft_grid.gather_from_grid(basis, dup_g_full, dup_g);
            fft_grid.gather_from_grid(basis, ddn_g_full, ddn_g);

            // 3. Compute spin sigma (|∇n_up|², ∇n_up·∇n_dn, |∇n_dn|²)
            auto spin_sigma = compute_spin_sigma(dup_g, ddn_g, basis, fft_grid);

            // 4. Evaluate spin-polarized GGA XC
            auto sgga = xc.evaluate_spin_gga(
                density_up_r, density_dn_r,
                spin_sigma.sigma_uu, spin_sigma.sigma_ud, spin_sigma.sigma_dd,
                volume);

            // 5. Compute GGA potential corrections
            RVec vgga_up(num_grid), vgga_dn(num_grid);
            compute_spin_gga_potential(
                dup_g, ddn_g,
                sgga.vsigma_uu, sgga.vsigma_ud, sgga.vsigma_dd,
                spin_sigma, basis, fft_grid,
                vgga_up, vgga_dn);

            // 6. Package into spin_xc_result with GGA corrections
            spin_xc_result.vxc_up.resize(num_grid);
            spin_xc_result.vxc_dn.resize(num_grid);
            for (int i = 0; i < num_grid; ++i) {
                spin_xc_result.vxc_up[i] = sgga.vxc_up[i] + vgga_up[i];
                spin_xc_result.vxc_dn[i] = sgga.vxc_dn[i] + vgga_dn[i];
            }
            e_xc = sgga.energy;
        } else if (nspin == 2) {
            // Spin-polarized LDA: separate V_xc for up and down
            spin_xc_result = xc.evaluate_spin(density_up_r, density_dn_r, volume);
            e_xc = spin_xc_result.energy;
        } else if (xc.is_gga()) {
            RVec sigma = compute_sigma(density_g, basis, fft_grid);
            xc_result = xc.evaluate_gga(density_r, sigma, volume);
            RVec vgga = compute_gga_potential(density_g, xc_result.vsigma, basis, fft_grid);
            for (int i = 0; i < num_grid; ++i) {
                xc_result.vxc[i] += vgga[i];
            }
            e_xc = xc_result.energy;
        } else {
            xc_result = xc.evaluate(density_r, volume);
            e_xc = xc_result.energy;
        }

        // d. Build V_eff(r) = V_H(r) + V_xc(r) + V_loc(r)
        //    V_H and V_loc are shared between spins; V_xc differs for nspin=2
        std::vector<complex_t> veff_g(num_grid);
        for (int i = 0; i < num_grid; ++i) {
            veff_g[i] = vh_full_g[i] + ng * vloc_full_g[i];
        }

        // Inverse FFT to get V_H + V_loc in real space
        std::vector<complex_t> vhl_r(num_grid);  // V_Hartree + V_local (shared)
        fft_grid.inverse(veff_g, vhl_r);

        // Build per-spin V_eff (or single V_eff for nspin=1)
        std::vector<complex_t> veff_r(num_grid);
        std::vector<complex_t> veff_up_r, veff_dn_r;

        if (nspin == 2) {
            veff_up_r.resize(num_grid);
            veff_dn_r.resize(num_grid);
            for (int i = 0; i < num_grid; ++i) {
                veff_up_r[i] = vhl_r[i] + spin_xc_result.vxc_up[i];
                veff_dn_r[i] = vhl_r[i] + spin_xc_result.vxc_dn[i];
            }
            // Use spin-up V_eff as default for converged_veff_r
            veff_r = veff_up_r;
        } else {
            for (int i = 0; i < num_grid; ++i) {
                veff_r[i] = vhl_r[i] + xc_result.vxc[i];
            }
        }

        // PAW: compute D_ij corrections and update NonlocalPP
        if (use_paw) {
            // FFT spin-averaged V_eff to G-space for D_ij integral
            std::vector<complex_t> veff_g_paw(num_grid);
            if (nspin == 2) {
                // Use spin-averaged V_eff for PAW D_ij
                std::vector<complex_t> veff_avg(num_grid);
                for (int i = 0; i < num_grid; ++i) {
                    veff_avg[i] = 0.5 * (veff_up_r[i] + veff_dn_r[i]);
                }
                fft_grid.forward(veff_avg, veff_g_paw);
            } else {
                fft_grid.forward(veff_r, veff_g_paw);
            }

            auto dij_paw = paw_calc.compute_dij_paw(veff_g_paw, grid_gcart, grid_g2, ecutrho);

            nonlocal_pp.reset_dij();
            for (size_t ia_paw = 0; ia_paw < dij_paw.size(); ++ia_paw) {
                if (ia_paw < paw_atom_to_nlpp_idx.size()) {
                    nonlocal_pp.add_dij_correction(paw_atom_to_nlpp_idx[ia_paw], dij_paw[ia_paw]);
                }
            }
        }

        // e. Solve eigenvalue problem at LOCAL k-points (per spin for nspin=2)
        //    Each MPI rank solves only its subset of k-points, then
        //    eigenvalues are gathered for the global Fermi level search.
        //
        //    For nspin=2: solve twice per k-point with V_eff_up and V_eff_dn
        //
        //    Local storage: indexed by local k-point index (0..nk_local-1)
        //    Global storage (eigenvalues only): indexed by global k-point index

        // Local wavefunctions/eigenvalues: [local_ik][band]
        std::vector<std::vector<double>> local_eigenvalues;
        std::vector<std::vector<CVec>> local_wavefunctions;

        // Per-spin local eigenvalues/wavefunctions for nspin=2
        // Layout: local_eigenvalues_spin[ispin][local_ik][ib]
        std::vector<std::vector<std::vector<double>>> local_eigenvalues_spin(nspin);
        std::vector<std::vector<std::vector<CVec>>> local_wavefunctions_spin(nspin);

        for (int ispin = 0; ispin < nspin; ++ispin) {
            if (nspin == 2) {
                ham.update_veff(ispin == 0 ? veff_up_r : veff_dn_r);
            } else {
                ham.update_veff(veff_r);
            }

            for (int iloc = 0; iloc < nk_local; ++iloc) {
                KRONOS_TIMER("eigensolver");
                int ik_global = my_kpoint_indices[iloc];
                auto h_apply = ham.get_apply_function(kpoints[ik_global]);
                auto precond = ham.kinetic_diagonal(kpoints[ik_global]);

                // Prepare k-point for nonlocal projectors (needed for PAW S operator)
                nonlocal_pp.prepare_kpoint(kpoints[ik_global]);

                // Construct S operator lambda for PAW (generalized eigenvalue)
                std::function<CVec(const CVec&)> s_apply_fn = nullptr;
                if (use_paw) {
                    s_apply_fn = [&](const CVec& psi) {
                        auto proj = nonlocal_pp.compute_projections(psi);
                        // Flatten beta projectors for PAW apply_s
                        std::vector<CVec> flat_beta;
                        for (size_t ia = 0; ia < nonlocal_pp.num_atoms(); ++ia) {
                            for (const auto& b : nonlocal_pp.cached_beta()[ia]) {
                                flat_beta.push_back(b);
                            }
                        }
                        // Build per-atom projections in UPF basis for PAW
                        // PAW works with UPF projector indices, not expanded
                        std::vector<std::vector<complex_t>> proj_per_atom;
                        for (size_t ia = 0; ia < nonlocal_pp.num_atoms(); ++ia) {
                            // For PAW, we pass the expanded projections directly
                            // since apply_s handles the q_ij in UPF space
                            proj_per_atom.push_back(proj[ia]);
                        }
                        return paw_calc.apply_s(psi, flat_beta, proj_per_atom);
                    };
                }

                // Wrap Hamiltonian with exact exchange when ACE is ready
                std::function<CVec(const CVec&)> h_apply_full;
                if (use_hybrid && exx->ace_ready()) {
                    int ik_ace = ik_global;
                    h_apply_full = [&h_apply, &exx, ik_ace](const CVec& psi) -> CVec {
                        CVec hpsi = h_apply(psi);
                        CVec vx = exx->apply_ace(psi, ik_ace);
                        for (size_t ig = 0; ig < hpsi.size(); ++ig) {
                            hpsi[ig] += vx[ig];
                        }
                        return hpsi;
                    };
                } else {
                    h_apply_full = h_apply;
                }

                EigenResult eigen = eigensolver.solve(h_apply_full, precond, num_bands, num_pw,
                                                      {}, s_apply_fn);

                if (nspin == 1) {
                    local_eigenvalues.push_back(eigen.eigenvalues);
                    local_wavefunctions.push_back(std::move(eigen.eigenvectors));
                }
                local_eigenvalues_spin[ispin].push_back(eigen.eigenvalues);
                local_wavefunctions_spin[ispin].push_back(std::move(eigen.eigenvectors));
            }
        }

        // Gather ALL eigenvalues from all ranks for the Fermi level search.
        // Each rank packs its local eigenvalues into a flat buffer, then we
        // allgather so every rank has the complete set.
        //
        // all_eigenvalues[global_ik][ib] -- full set for nspin=1
        // all_eigenvalues_spin[ispin][global_ik][ib] -- full set for nspin=2
        std::vector<std::vector<double>> all_eigenvalues(nk_total);
        std::vector<std::vector<std::vector<double>>> all_eigenvalues_spin(nspin,
            std::vector<std::vector<double>>(nk_total));

        for (int ispin = 0; ispin < nspin; ++ispin) {
            // Pack local eigenvalues into flat send buffer
            std::vector<double> sendbuf(nk_local * num_bands, 0.0);
            for (int iloc = 0; iloc < nk_local; ++iloc) {
                for (int ib = 0; ib < num_bands; ++ib) {
                    sendbuf[iloc * num_bands + ib] =
                        (nspin == 1) ? local_eigenvalues[iloc][ib]
                                     : local_eigenvalues_spin[ispin][iloc][ib];
                }
            }

            // Prepare recvcounts and displacements for allgatherv
            std::vector<int> recvcounts(mpi_size);
            std::vector<int> displs(mpi_size);
            for (int r = 0; r < mpi_size; ++r) {
                int nk_r = 0;
                for (int ik = 0; ik < nk_total; ++ik) {
                    if (ik % mpi_size == r) nk_r++;
                }
                recvcounts[r] = nk_r * num_bands;
            }
            displs[0] = 0;
            for (int r = 1; r < mpi_size; ++r) {
                displs[r] = displs[r - 1] + recvcounts[r - 1];
            }
            int total_recv = displs[mpi_size - 1] + recvcounts[mpi_size - 1];
            std::vector<double> recvbuf(total_recv, 0.0);

            mpi::allgatherv(sendbuf.data(), nk_local * num_bands,
                            recvbuf.data(), recvcounts.data(), displs.data());

            // Unpack into all_eigenvalues / all_eigenvalues_spin
            for (int r = 0; r < mpi_size; ++r) {
                int offset = displs[r];
                int local_idx = 0;
                for (int ik = 0; ik < nk_total; ++ik) {
                    if (ik % mpi_size != r) continue;
                    std::vector<double> evals(num_bands);
                    for (int ib = 0; ib < num_bands; ++ib) {
                        evals[ib] = recvbuf[offset + local_idx * num_bands + ib];
                    }
                    all_eigenvalues_spin[ispin][ik] = evals;
                    if (nspin == 1) {
                        all_eigenvalues[ik] = evals;
                    }
                    local_idx++;
                }
            }
        }

        // f. Find Fermi level and occupations using GLOBAL eigenvalues.
        //    For nspin=2: combine eigenvalues from both spins into a single
        //    Fermi search with spin_factor=1 and doubled k-point weights.
        //    Layout: [spin_up k0, spin_up k1, ..., spin_dn k0, spin_dn k1, ...]
        FermiResult fermi;
        std::vector<FermiResult> fermi_spin(nspin);

        if (nspin == 2) {
            // Build combined eigenvalue/weight arrays
            std::vector<std::vector<double>> combined_evals;
            std::vector<double> combined_weights;
            for (int ispin = 0; ispin < nspin; ++ispin) {
                for (int ik = 0; ik < nk_total; ++ik) {
                    combined_evals.push_back(all_eigenvalues_spin[ispin][ik]);
                    combined_weights.push_back(kweights[ik]);
                }
            }
            // spin_factor=1 for spin-polarized, each spin channel counted once
            fermi = FermiSolver::find_fermi_level(
                combined_evals, combined_weights, target_electrons,
                calc_params_.smearing, calc_params_.degauss, 1);

            // Split occupations back to per-spin (global indexing)
            for (int ispin = 0; ispin < nspin; ++ispin) {
                fermi_spin[ispin].occupations.resize(nk_total);
                for (int ik = 0; ik < nk_total; ++ik) {
                    size_t idx = ispin * nk_total + ik;
                    fermi_spin[ispin].occupations[ik] = fermi.occupations[idx];
                }
                fermi_spin[ispin].fermi_energy = fermi.fermi_energy;
            }

        } else {
            fermi = FermiSolver::find_fermi_level(
                all_eigenvalues, kweights, target_electrons,
                calc_params_.smearing, calc_params_.degauss, spin_factor);
        }

        // g. Compute new density from LOCAL k-point wavefunctions, then
        //    allreduce to get the total density across all ranks.
        RVec density_out(num_grid, 0.0);
        RVec density_out_up(num_grid, 0.0), density_out_dn(num_grid, 0.0);

        // Extract local occupations from global Fermi result
        std::vector<std::vector<double>> local_occupations(nk_local);
        std::vector<std::vector<std::vector<double>>> local_occupations_spin(nspin,
            std::vector<std::vector<double>>(nk_local));
        for (int iloc = 0; iloc < nk_local; ++iloc) {
            int ik_global = my_kpoint_indices[iloc];
            if (nspin == 1) {
                local_occupations[iloc] = fermi.occupations[ik_global];
            }
            for (int ispin = 0; ispin < nspin; ++ispin) {
                if (nspin == 2) {
                    local_occupations_spin[ispin][iloc] =
                        fermi_spin[ispin].occupations[ik_global];
                }
            }
        }

        // PAW: collect projections and compute ρ_ij after occupations are known
        if (use_paw) {
            // Build local kweights for PAW
            std::vector<double> paw_kweights(nk_local);
            for (int iloc = 0; iloc < nk_local; ++iloc) {
                paw_kweights[iloc] = kweights[my_kpoint_indices[iloc]];
            }

            // Build projections: per-k-point, per-band, per-atom (UPF indices)
            std::vector<std::vector<std::vector<complex_t>>> projections_paw;

            for (int iloc = 0; iloc < nk_local; ++iloc) {
                int ik_global = my_kpoint_indices[iloc];
                nonlocal_pp.prepare_kpoint(kpoints[ik_global]);

                std::vector<std::vector<complex_t>> band_projs;
                for (int ib = 0; ib < num_bands; ++ib) {
                    const auto& wfc = (nspin == 1)
                        ? local_wavefunctions[iloc][ib]
                        : local_wavefunctions_spin[0][iloc][ib];

                    auto expanded_proj = nonlocal_pp.compute_projections(wfc);

                    // Collapse expanded projections to UPF indices for each PAW atom.
                    // PAW compute_rho_ij expects proj[ip] where ip is UPF projector index.
                    // For each UPF projector, take the m=0 expanded projection.
                    std::vector<complex_t> flat_upf_proj;
                    for (size_t ia_paw = 0; ia_paw < paw_atom_to_nlpp_idx.size(); ++ia_paw) {
                        size_t ia_nlpp = paw_atom_to_nlpp_idx[ia_paw];
                        int nproj_upf = nonlocal_pp.num_upf_projectors(ia_nlpp);
                        const auto& emap = nonlocal_pp.expanded_map(ia_nlpp);
                        const auto& atom_proj = expanded_proj[ia_nlpp];

                        for (int ip_upf = 0; ip_upf < nproj_upf; ++ip_upf) {
                            complex_t proj_val{0.0, 0.0};
                            for (size_t ie = 0; ie < emap.size(); ++ie) {
                                if (emap[ie].upf_beta_index == ip_upf && emap[ie].m == 0) {
                                    proj_val = atom_proj[ie];
                                    break;
                                }
                            }
                            flat_upf_proj.push_back(proj_val);
                        }
                    }
                    band_projs.push_back(std::move(flat_upf_proj));
                }
                projections_paw.push_back(std::move(band_projs));
            }

            // Compute ρ_ij using occupations
            const auto& occs_for_paw = (nspin == 1)
                ? local_occupations : local_occupations_spin[0];
            paw_calc.compute_rho_ij(projections_paw, occs_for_paw, paw_kweights, spin_factor);
        }

        auto accumulate_density = [&](const std::vector<std::vector<CVec>>& wfcs,
                                       const std::vector<std::vector<double>>& occs,
                                       RVec& dens_out) {
            for (int iloc = 0; iloc < nk_local; ++iloc) {
                int ik_global = my_kpoint_indices[iloc];
                auto ke_k = basis.kinetic_energies(kpoints[ik_global]);
                for (int n = 0; n < num_bands; ++n) {
                    double occ = kweights[ik_global] * occs[iloc][n];
                    if (occ < 1e-12) continue;

                    CVec psi_masked(num_pw);
                    for (int ig = 0; ig < num_pw; ++ig) {
                        psi_masked[ig] = (ke_k[ig] <= calc_params_.ecutwfc + 1.0e-6)
                                       ? wfcs[iloc][n][ig]
                                       : complex_t{0.0, 0.0};
                    }
                    std::vector<complex_t> psi_grid(num_grid, complex_t{0.0, 0.0});
                    fft_grid.scatter_to_grid(basis, psi_masked, psi_grid);
                    std::vector<complex_t> psi_r_vec(num_grid);
                    fft_grid.inverse(psi_grid, psi_r_vec);
                    for (int i = 0; i < num_grid; ++i) {
                        dens_out[i] += occ * std::norm(psi_r_vec[i]);
                    }
                }
            }
        };

        if (nspin == 2) {
            accumulate_density(local_wavefunctions_spin[0], local_occupations_spin[0], density_out_up);
            accumulate_density(local_wavefunctions_spin[1], local_occupations_spin[1], density_out_dn);
            for (int i = 0; i < num_grid; ++i) {
                density_out[i] = density_out_up[i] + density_out_dn[i];
            }
        } else {
            accumulate_density(local_wavefunctions, local_occupations, density_out);
        }

        // MPI: allreduce partial density contributions from all ranks
        mpi::allreduce_sum_inplace(density_out.data(), num_grid);
        if (nspin == 2) {
            mpi::allreduce_sum_inplace(density_out_up.data(), num_grid);
            mpi::allreduce_sum_inplace(density_out_dn.data(), num_grid);
        }

        // Normalize: integral n(r) dr = N_electrons
        double dn_sum = 0.0;
        for (int i = 0; i < num_grid; ++i) {
            dn_sum += density_out[i];
        }
        if (dn_sum > 1e-15) {
            double scale = target_electrons / (dn_sum * volume / num_grid);
            for (int i = 0; i < num_grid; ++i) {
                density_out[i] *= scale;
            }
            if (nspin == 2) {
                for (int i = 0; i < num_grid; ++i) {
                    density_out_up[i] *= scale;
                    density_out_dn[i] *= scale;
                }
            }
        }

        // h. Compute energies on the full density grid.
        //    E_H = (Ω/2) Σ_G conj(V_H_FFT(G)) × n_FFT(G) / N²
        //    E_loc = Ω × Σ_G conj(V_loc_phys(G)) × n_FFT(G) / N
        double e_hartree = 0.0;
        for (int i = 0; i < num_grid; ++i) {
            e_hartree += std::real(std::conj(vh_full_g[i]) * density_g_full[i]);
        }
        e_hartree *= 0.5 * volume / (ng * ng);

        // e_xc was set above in section c

        double e_local = 0.0;
        for (int i = 0; i < num_grid; ++i) {
            e_local += std::real(std::conj(vloc_full_g[i]) * density_g_full[i]);
        }
        e_local *= volume / ng;

        // Band energy: for nspin=2, sum over both spin channels
        double e_band = 0.0;
        if (nspin == 2) {
            for (int ispin = 0; ispin < nspin; ++ispin) {
                e_band += compute_band_energy(all_eigenvalues_spin[ispin],
                                              fermi_spin[ispin].occupations, kweights);
            }
        } else {
            e_band = compute_band_energy(all_eigenvalues, fermi.occupations, kweights);
        }

        // Total energy = E_band - E_H + E_xc - integral(V_xc * n)
        // (double counting correction)
        double vxc_integral = 0.0;
        if (nspin == 2) {
            // V_xc integral = integral(V_xc_up * n_up + V_xc_dn * n_dn)
            for (int i = 0; i < num_grid; ++i) {
                vxc_integral += spin_xc_result.vxc_up[i] * density_up_r[i]
                              + spin_xc_result.vxc_dn[i] * density_dn_r[i];
            }
        } else {
            for (int i = 0; i < num_grid; ++i) {
                vxc_integral += xc_result.vxc[i] * density_r[i];
            }
        }
        vxc_integral *= volume / num_grid;

        double total_e = e_band - e_hartree + e_xc - vxc_integral;

        // PAW one-center energy correction
        double e_paw = 0.0;
        if (use_paw) {
            e_paw = paw_calc.one_center_energy();
            total_e += e_paw;
        }

        // Exact exchange energy for hybrid functionals
        double e_exx = 0.0;
        if (use_hybrid && exx->ace_ready()) {
            // For serial (mpi_size==1), local == global.
            // MPI + hybrid not supported in v0.8.
            std::vector<Vec3> exx_kpts(nk_local);
            std::vector<double> exx_kwts(nk_local);
            for (int iloc = 0; iloc < nk_local; ++iloc) {
                int ik_global = my_kpoint_indices[iloc];
                exx_kpts[iloc] = kpoints[ik_global];
                exx_kwts[iloc] = kweights[ik_global];
            }
            e_exx = exx->exchange_energy(
                (nspin == 1) ? local_wavefunctions : local_wavefunctions_spin[0],
                (nspin == 1) ? local_occupations : local_occupations_spin[0],
                exx_kpts, exx_kwts);
            total_e += e_exx;
        }

        // Smearing entropy correction (Gaussian: -T*S = -0.5*sigma/sqrt(pi) * sum exp(-x^2))
        // QE uses w0gauss(x) = -0.5*exp(-x^2)/sqrt(pi) per state.
        double e_smearing = 0.0;
        if (calc_params_.smearing == SmearingType::Gaussian && calc_params_.degauss > 0) {
            const double deg = calc_params_.degauss;
            const double ef = fermi.fermi_energy;
            for (int ispin = 0; ispin < nspin; ++ispin) {
                const auto& evals = (nspin == 2) ? all_eigenvalues_spin[ispin] : all_eigenvalues;
                for (size_t ik = 0; ik < kpoints.size(); ++ik) {
                    for (int n = 0; n < num_bands; ++n) {
                        double x = (evals[ik][n] - ef) / deg;
                        e_smearing += kweights[ik] * std::exp(-x * x);
                    }
                }
            }
            // For nspin=1 (unpolarized), the spin_factor=2 is implicit
            // For nspin=2, we summed both spin channels explicitly
            if (nspin == 1) e_smearing *= 2.0;
            e_smearing *= -0.5 * deg / std::sqrt(constants::pi);
        }
        total_e += e_smearing;

        // Check convergence
        double de = (step == 1) ? 0.0 : std::abs(total_e - prev_energy);

        // Compute density change norm in G-space (PW components only).
        // This avoids aliasing artifacts from the real-space grid and
        // converges cleanly with the PW basis.
        double dn = 0.0;
        {
            // FFT the density difference to G-space
            std::vector<complex_t> diff_c(num_grid);
            for (int i = 0; i < num_grid; ++i) {
                diff_c[i] = complex_t{density_out[i] - density_r[i], 0.0};
            }
            std::vector<complex_t> diff_g_full(num_grid);
            fft_grid.forward(diff_c, diff_g_full);

            // Gather PW coefficients and compute L2 norm
            CVec diff_g_pw(num_pw);
            fft_grid.gather_from_grid(basis, diff_g_full, diff_g_pw);

            double sum2 = 0.0;
            for (int ig = 0; ig < num_pw; ++ig) {
                sum2 += std::norm(diff_g_pw[ig]);  // |Δn(G)|²
            }
            dn = std::sqrt(sum2) / num_grid;  // normalize by grid size (FFT convention)
        }
        if (target_electrons > 1e-15) {
            dn /= target_electrons;  // relative density change
        }

        // Compute kinetic and nonlocal energies from band decomposition.
        // Only iterate over LOCAL k-points (we have wavefunctions only for
        // our subset), then allreduce to get the global nonlocal energy.
        double e_nonlocal = 0.0;
        if (nspin == 2) {
            for (int ispin = 0; ispin < nspin; ++ispin) {
                for (int iloc = 0; iloc < nk_local; ++iloc) {
                    int ik_global = my_kpoint_indices[iloc];
                    for (int n = 0; n < num_bands; ++n) {
                        double occ = kweights[ik_global] * local_occupations_spin[ispin][iloc][n];
                        if (occ < 1e-12) continue;
                        double enl = nonlocal_pp.energy(
                            {local_wavefunctions_spin[ispin][iloc][n]}, {1.0}, kpoints[ik_global]);
                        e_nonlocal += occ * enl;
                    }
                }
            }
        } else {
            for (int iloc = 0; iloc < nk_local; ++iloc) {
                int ik_global = my_kpoint_indices[iloc];
                for (int n = 0; n < num_bands; ++n) {
                    double occ = kweights[ik_global] * local_occupations[iloc][n];
                    if (occ < 1e-12) continue;
                    double enl = nonlocal_pp.energy(
                        {local_wavefunctions[iloc][n]}, {1.0}, kpoints[ik_global]);
                    e_nonlocal += occ * enl;
                }
            }
        }
        // MPI: allreduce partial nonlocal energy
        mpi::allreduce_sum_inplace(&e_nonlocal, 1);
        double e_kinetic = e_band - 2.0 * e_hartree - e_local - vxc_integral - e_nonlocal;

        result.total_energy_ry = total_e;
        prev_energy = total_e;
        result.scf_steps = step;
        result.kinetic_energy = e_kinetic;
        result.hartree_energy = e_hartree;
        result.xc_energy = e_xc;
        result.local_pp_energy = e_local;
        result.nonlocal_pp_energy = e_nonlocal;
        result.smearing_energy = e_smearing;
        result.paw_energy = e_paw;
        result.exx_energy = e_exx;
        result.fermi_energy_ev = fermi.fermi_energy * constants::rydberg_to_ev;

        // Set eigenvalues
        if (nspin == 2) {
            // Store spin-up eigenvalues as the default
            result.eigenvalues = all_eigenvalues_spin[0];
            result.eigenvalues_spin = all_eigenvalues_spin;
            // Compute magnetization
            double mag_total = 0.0, mag_abs = 0.0;
            for (int i = 0; i < num_grid; ++i) {
                double m = density_out_up[i] - density_out_dn[i];
                mag_total += m;
                mag_abs += std::abs(m);
            }
            mag_total *= volume / num_grid;
            mag_abs *= volume / num_grid;
            result.total_magnetization = mag_total;
            result.absolute_magnetization = mag_abs;
        } else {
            result.eigenvalues = all_eigenvalues;
        }

        // Retain data for post-SCF force calculation.
        // We store LOCAL wavefunctions/occupations indexed by local k-point.
        // The force calculation will iterate only over local k-points and
        // allreduce the result.
        if (nspin == 2) {
            // Combine both spin channels' wavefunctions/occupations
            converged_wavefunctions.clear();
            converged_occupations.clear();
            for (int iloc = 0; iloc < nk_local; ++iloc) {
                // Concatenate spin-up and spin-down at each local k-point
                std::vector<CVec> combined_wfcs;
                std::vector<double> combined_occs;
                for (int ispin = 0; ispin < nspin; ++ispin) {
                    for (int n = 0; n < num_bands; ++n) {
                        combined_wfcs.push_back(local_wavefunctions_spin[ispin][iloc][n]);
                        combined_occs.push_back(local_occupations_spin[ispin][iloc][n]);
                    }
                }
                converged_wavefunctions.push_back(std::move(combined_wfcs));
                converged_occupations.push_back(std::move(combined_occs));
            }
        } else {
            converged_wavefunctions = local_wavefunctions;
            converged_occupations = local_occupations;
        }
        converged_density_g = density_g;
        converged_density_g_full = density_g_full;
        converged_veff_r = veff_r;
        converged_exc_energy = e_xc;
        if (nspin == 2) {
            // For stress XC: store total vxc as weighted average
            converged_vxc_r.resize(num_grid);
            for (int i = 0; i < num_grid; ++i) {
                converged_vxc_r[i] = 0.5 * (spin_xc_result.vxc_up[i] + spin_xc_result.vxc_dn[i]);
            }
        } else {
            converged_vxc_r = xc_result.vxc;
        }
        converged_density_r = density_r;

        auto step_end = std::chrono::high_resolution_clock::now();
        double wall = std::chrono::duration<double>(step_end - step_start).count();
        if (is_root) {
            print_scf_step(step, total_e, de, dn, wall);
            if (nspin == 2) {
                std::printf("         mag = %.4f muB\n", result.total_magnetization);
            }
        }

        // Convergence check: energy convergence is the primary criterion.
        // Density convergence in real space can stall due to high-G components
        // that don't affect the energy (beyond the PW cutoff).
        if (step > 1 && de < conv_params_.energy_threshold) {
            if (dn < conv_params_.density_threshold) {
                result.converged = true;
                break;
            }
            // If energy converged but density hasn't after many steps,
            // declare convergence based on energy alone
            if (de < conv_params_.energy_threshold * 0.01) {
                result.converged = true;
                break;
            }
        }

        // Hard guardrail: abort if energy oscillates wildly
        // Skip early steps where large energy changes are normal,
        // especially with real pseudopotentials where initial convergence
        // can be slow.
        if (step > 15 && de > 1.0) {  // > 1 Ry oscillation
            logger.error("scf", "Energy oscillation > 1 Ry, aborting",
                {{"de", std::to_string(de)}});
            break;
        }

        // Symmetrize the output density before mixing.
        // When using IBZ k-points, the density from wavefunctions lacks full
        // crystal symmetry. Symmetrizing it ensures the mixer converges to
        // a density with the correct symmetry, matching the QE convention.
        //
        // For Gamma-only (1 k-point), skip symmetrization because the
        // density from Gamma wavefunctions already has full crystal symmetry.
        //
        // We do NOT clamp negative values after symmetrization because the
        // non-linear clamp breaks the self-consistency condition and can
        // cause DIIS instability. Tiny negative values from the FFT
        // round-trip are harmless and within numerical precision.
#ifdef KRONOS_HAS_SPGLIB
        if (kpoints.size() > 1) {
            // FFT density_out to G-space, symmetrize, IFFT back
            std::vector<complex_t> dout_c(num_grid);
            for (int i = 0; i < num_grid; ++i) {
                dout_c[i] = complex_t{density_out[i], 0.0};
            }
            std::vector<complex_t> dout_g(num_grid);
            fft_grid.forward(dout_c, dout_g);
            symmetrize_density_g(crystal_, dout_g, grid_dims);
            fft_grid.inverse(dout_g, dout_c);
            for (int i = 0; i < num_grid; ++i) {
                density_out[i] = std::real(dout_c[i]);
            }

            // For nspin=2, also symmetrize per-spin densities
            if (nspin == 2) {
                // Symmetrize density_out_up
                for (int i = 0; i < num_grid; ++i) {
                    dout_c[i] = complex_t{density_out_up[i], 0.0};
                }
                fft_grid.forward(dout_c, dout_g);
                symmetrize_density_g(crystal_, dout_g, grid_dims);
                fft_grid.inverse(dout_g, dout_c);
                for (int i = 0; i < num_grid; ++i) {
                    density_out_up[i] = std::real(dout_c[i]);
                }

                // Symmetrize density_out_dn
                for (int i = 0; i < num_grid; ++i) {
                    dout_c[i] = complex_t{density_out_dn[i], 0.0};
                }
                fft_grid.forward(dout_c, dout_g);
                symmetrize_density_g(crystal_, dout_g, grid_dims);
                fft_grid.inverse(dout_g, dout_c);
                for (int i = 0; i < num_grid; ++i) {
                    density_out_dn[i] = std::real(dout_c[i]);
                }

                // Recompute total from symmetrized per-spin densities
                for (int i = 0; i < num_grid; ++i) {
                    density_out[i] = density_out_up[i] + density_out_dn[i];
                }
            }
        }
#endif

        // i. Mix densities (with optional Kerker preconditioning for metals)
        //    For nspin=2, mix total density and magnetization separately.
        //    This is more robust than mixing n_up/n_dn independently because
        //    the total density and magnetization have different convergence scales.
        if (nspin == 2) {
            // Compute magnetization density: m = n_up - n_dn
            RVec mag_in(num_grid), mag_out(num_grid);
            for (int i = 0; i < num_grid; ++i) {
                mag_in[i]  = density_up_r[i] - density_dn_r[i];
                mag_out[i] = density_out_up[i] - density_out_dn[i];
            }

            // Mix total density with main Pulay mixer
            density_r = mixer.mix(density_r, density_out);

            // Mix magnetization with separate (conservative) mixer
            RVec mag_mixed = mixer_mag.mix(mag_in, mag_out);

            // Reconstruct per-spin densities from (n, m)
            for (int i = 0; i < num_grid; ++i) {
                density_up_r[i] = std::max(0.5 * (density_r[i] + mag_mixed[i]), 0.0);
                density_dn_r[i] = std::max(0.5 * (density_r[i] - mag_mixed[i]), 0.0);
            }
        } else if (use_kerker) {
            RVec residual_r(num_grid);
            for (int i = 0; i < num_grid; ++i) {
                residual_r[i] = density_out[i] - density_r[i];
            }
            std::vector<complex_t> res_c(num_grid);
            for (int i = 0; i < num_grid; ++i) {
                res_c[i] = complex_t{residual_r[i], 0.0};
            }
            std::vector<complex_t> res_g_full(num_grid);
            fft_grid.forward(res_c, res_g_full);
            CVec res_g_pw(num_pw);
            fft_grid.gather_from_grid(basis, res_g_full, res_g_pw);

            CVec res_kerker = kerker.apply(res_g_pw, g_norm2_pw);

            std::vector<complex_t> res_kerker_grid(num_grid, complex_t{0.0, 0.0});
            fft_grid.scatter_to_grid(basis, res_kerker, res_kerker_grid);
            std::vector<complex_t> res_kerker_r(num_grid);
            fft_grid.inverse(res_kerker_grid, res_kerker_r);

            RVec density_out_precond(num_grid);
            for (int i = 0; i < num_grid; ++i) {
                density_out_precond[i] = density_r[i] + std::real(res_kerker_r[i]);
            }
            density_r = mixer.mix(density_r, density_out_precond);
        } else {
            density_r = mixer.mix(density_r, density_out);
        }

        // Clamp negative density and renormalize to conserve charge
        if (nspin != 2) {  // nspin=2 already handled above
            double neg_charge = 0.0;
            for (int i = 0; i < num_grid; ++i) {
                if (density_r[i] < 0.0) {
                    neg_charge += -density_r[i];
                    density_r[i] = 0.0;
                }
            }
            if (neg_charge > 1e-15) {
                double pos_sum = 0.0;
                for (int i = 0; i < num_grid; ++i) {
                    pos_sum += density_r[i];
                }
                if (pos_sum > 1e-15) {
                    double rescale = (pos_sum + neg_charge) / pos_sum;
                    for (int i = 0; i < num_grid; ++i) {
                        density_r[i] *= rescale;
                    }
                }
            }
        }
    }

    // Outer loop convergence check for hybrid functionals
    if (use_hybrid) {
        double de = std::abs(result.total_energy_ry - outer_energy_prev);
        if (outer > 0 && de < conv_params_.energy_threshold * 10.0) {
            if (is_root) {
                std::printf("  Hybrid outer SCF converged: dE = %.2e Ry\n", de);
            }
            break;
        }
        outer_energy_prev = result.total_energy_ry;
    }
    } // end outer loop

    // ----------------------------------------------------------------
    // 7. Post-SCF: Ewald energy and Hellmann-Feynman forces
    // ----------------------------------------------------------------
    {
        KRONOS_TIMER("ewald");
        auto ewald_result = EwaldCalculator::compute(crystal_, pseudopotentials_);
        result.ewald_energy = ewald_result.energy;
        result.ewald_forces = std::move(ewald_result.forces);

        // Add Ewald ion-ion energy to the total energy
        result.total_energy_ry += result.ewald_energy;

        logger.info("ewald", "Ewald summation completed",
            {{"ewald_energy_ry", std::to_string(result.ewald_energy)}});
    }

    // Compute Hellmann-Feynman forces if the calculation type requires them
    // (SCF + Relax both need forces; Bands and DOS do not)
    const bool compute_forces = (calc_params_.type == CalculationType::SCF
                              || calc_params_.type == CalculationType::Relax
                              || calc_params_.type == CalculationType::VCRelax);

    // Build local k-points/weights arrays for post-SCF calculations.
    // Needed by force calculator when using MPI k-point parallelization.
    std::vector<Vec3> local_kpoints(nk_local);
    std::vector<double> local_kweights(nk_local);
    for (int iloc = 0; iloc < nk_local; ++iloc) {
        int ik_global = my_kpoint_indices[iloc];
        local_kpoints[iloc] = kpoints[ik_global];
        local_kweights[iloc] = kweights[ik_global];
    }

    if (compute_forces && !converged_density_g_full.empty()) {
        KRONOS_TIMER("forces");

        // Local pseudopotential forces (must use full density grid to match energy)
        // These do not depend on k-points, so all ranks compute the same result.
        result.local_forces = ForceCalculator::compute_local_forces(
            crystal_, pseudopotentials_, converged_density_g_full,
            grid_gcart, grid_g2, ecutrho, volume, num_grid);

        // Nonlocal pseudopotential forces: each rank computes partial forces
        // from its LOCAL k-points only, then allreduce to sum across ranks.
        result.nonlocal_forces = ForceCalculator::compute_nonlocal_forces(
            crystal_, basis, pseudopotentials_,
            converged_wavefunctions, converged_occupations,
            local_kpoints, local_kweights, spin_factor);

        // MPI: allreduce nonlocal forces (partial sums from each rank)
        {
            const size_t natoms = crystal_.num_atoms();
            std::vector<double> nl_forces_flat(natoms * 3);
            for (size_t ia = 0; ia < natoms; ++ia) {
                nl_forces_flat[ia * 3 + 0] = result.nonlocal_forces[ia][0];
                nl_forces_flat[ia * 3 + 1] = result.nonlocal_forces[ia][1];
                nl_forces_flat[ia * 3 + 2] = result.nonlocal_forces[ia][2];
            }
            mpi::allreduce_sum_inplace(nl_forces_flat.data(),
                                       static_cast<int>(natoms * 3));
            for (size_t ia = 0; ia < natoms; ++ia) {
                result.nonlocal_forces[ia][0] = nl_forces_flat[ia * 3 + 0];
                result.nonlocal_forces[ia][1] = nl_forces_flat[ia * 3 + 1];
                result.nonlocal_forces[ia][2] = nl_forces_flat[ia * 3 + 2];
            }
        }

        // PAW augmentation force correction
        if (use_paw) {
            // Build V_eff in G-space for PAW force computation
            std::vector<complex_t> veff_g_paw(num_grid);
            fft_grid.forward(converged_veff_r, veff_g_paw);
            result.paw_forces = paw_calc.compute_paw_forces(
                veff_g_paw, grid_gcart, grid_g2, ecutrho);
        }

        // Total forces: Ewald + local + nonlocal (+ PAW if present)
        result.forces = ForceCalculator::compute_total_forces(
            result.ewald_forces, result.local_forces, result.nonlocal_forces);

        // Add PAW force contribution
        if (use_paw && !result.paw_forces.empty()) {
            for (size_t ia = 0; ia < crystal_.num_atoms(); ++ia) {
                for (int d = 0; d < 3; ++d) {
                    result.forces[ia][d] += result.paw_forces[ia][d];
                }
            }
        }

#ifdef KRONOS_HAS_SPGLIB
        // Symmetrize forces using crystal point group operations.
        // Raw forces from IBZ k-points don't have full crystal symmetry
        // because different k-points have different active PW counts.
        auto forces_unsym = result.forces;
        result.forces = symmetrize_forces(crystal_, result.forces);
#endif

        // Print force summary (rank 0 only)
        if (is_root) {
            std::printf("\nHellmann-Feynman forces (Ry/bohr):\n");
            std::printf("  Atom  Symbol      Fx            Fy            Fz\n");
            for (size_t ia = 0; ia < crystal_.num_atoms(); ++ia) {
                std::printf("  %3zu   %2s     %12.8f  %12.8f  %12.8f\n",
                            ia + 1,
                            crystal_.atom(ia).symbol.c_str(),
                            result.forces[ia][0],
                            result.forces[ia][1],
                            result.forces[ia][2]);
            }

            // Print maximum force magnitude
            double max_force = 0.0;
            for (size_t ia = 0; ia < crystal_.num_atoms(); ++ia) {
                double f2 = result.forces[ia][0] * result.forces[ia][0]
                          + result.forces[ia][1] * result.forces[ia][1]
                          + result.forces[ia][2] * result.forces[ia][2];
                max_force = std::max(max_force, std::sqrt(f2));
            }
            std::printf("  Max |F| = %.8f Ry/bohr\n", max_force);

            logger.info("forces", "Hellmann-Feynman forces computed",
                {{"max_force_ry_bohr", std::to_string(max_force)},
                 {"num_atoms", std::to_string(crystal_.num_atoms())}});
        }
    }

    // ----------------------------------------------------------------
    // 8. Post-SCF: Stress tensor
    // ----------------------------------------------------------------
    if (compute_forces && !converged_density_g_full.empty()) {
        KRONOS_TIMER("stress");

        result.stress_kinetic = StressCalculator::compute_kinetic_stress(
            crystal_, basis,
            converged_wavefunctions, converged_occupations,
            local_kpoints, local_kweights);

        result.stress_hartree = StressCalculator::compute_hartree_stress(
            converged_density_g_full, grid_gcart, grid_g2,
            ecutrho, volume, num_grid);

        result.stress_xc = StressCalculator::compute_xc_stress(
            converged_exc_energy, converged_vxc_r, converged_density_r,
            volume, num_grid);

        result.stress_local = StressCalculator::compute_local_stress(
            crystal_, pseudopotentials_,
            converged_density_g_full, vloc_full_g,
            grid_gcart, grid_g2, ecutrho, volume, num_grid);

        result.stress_nonlocal = StressCalculator::compute_nonlocal_stress(
            crystal_, basis, pseudopotentials_,
            converged_wavefunctions, converged_occupations,
            local_kpoints, local_kweights);

        result.stress_ewald = StressCalculator::compute_ewald_stress(
            crystal_, pseudopotentials_);

        result.stress = StressCalculator::compute_total_stress(
            result.stress_kinetic, result.stress_hartree, result.stress_xc,
            result.stress_local, result.stress_nonlocal, result.stress_ewald);

        // Add PAW stress correction
        if (use_paw) {
            std::vector<complex_t> veff_g_stress(num_grid);
            fft_grid.forward(converged_veff_r, veff_g_stress);
            Mat3 paw_stress = paw_calc.compute_paw_stress(
                veff_g_stress, grid_gcart, grid_g2, ecutrho);
            for (int a = 0; a < 3; ++a) {
                for (int b = 0; b < 3; ++b) {
                    result.stress[a][b] += paw_stress[a][b];
                }
            }
        }

        result.pressure_gpa = StressCalculator::pressure_gpa(result.stress);

        if (is_root) {
            std::printf("\nStress tensor (Ry/bohr^3):\n");
            for (int a = 0; a < 3; ++a) {
                std::printf("  %14.8f %14.8f %14.8f\n",
                            result.stress[a][0], result.stress[a][1], result.stress[a][2]);
            }
            std::printf("  Pressure = %.4f GPa\n", result.pressure_gpa);

            logger.info("stress", "Stress tensor computed",
                {{"pressure_gpa", std::to_string(result.pressure_gpa)}});
        }
    }

    // Store the converged effective potential for band structure calculations
    result.converged_veff_r = std::move(converged_veff_r);

    result.total_energy_ev = result.total_energy_ry * constants::rydberg_to_ev;
    result.timing = TimerRegistry::instance().as_map();

    if (is_root) {
        if (result.converged) {
            std::printf("\nCONVERGED in %d steps. Total energy: %.6f Ry (including Ewald: %.6f Ry)\n",
                        result.scf_steps, result.total_energy_ry, result.ewald_energy);
        } else {
            std::printf("\nNOT CONVERGED after %d steps. Total energy: %.6f Ry\n",
                        result.scf_steps, result.total_energy_ry);
        }
        if (nspin == 2) {
            std::printf("  Magnetization: total = %.4f muB, absolute = %.4f muB\n",
                        result.total_magnetization, result.absolute_magnetization);
        }
    }

    return result;
}

} // namespace kronos
