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
#include "solver/davidson.hpp"
#include "solver/mixing.hpp"
#include "solver/fermi.hpp"
#include "core/constants.hpp"
#include "utils/timer.hpp"
#include "utils/logger.hpp"
#include "utils/radial_integral.hpp"
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

    // If no pseudopotentials have rho_atomic data, fall back to uniform density
    bool have_rho_atomic = false;
    for (const auto& [symbol, pp] : pseudopotentials_) {
        if (!pp.rho_atomic.empty()) {
            have_rho_atomic = true;
            break;
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

    // 1. Generate k-points first (needed to determine basis expansion)
    auto kpoint_data = KPointGenerator::generate_monkhorst_pack(
        calc_params_.kpoints, crystal_);
    std::vector<Vec3> kpoints = kpoint_data.kpoints;
    std::vector<double> kweights = kpoint_data.weights;

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
         {"num_kpoints", std::to_string(kpoints.size())}});

    // Log active PW count per k-point (G-vectors where |k+G|^2 <= ecutwfc)
    for (size_t ik = 0; ik < kpoints.size(); ++ik) {
        auto ke = basis.kinetic_energies(kpoints[ik]);
        int active = 0;
        for (size_t ig = 0; ig < ke.size(); ++ig) {
            if (ke[ig] <= calc_params_.ecutwfc + 1.0e-6) active++;
        }
        std::printf("  k-point %zu: (%7.4f %7.4f %7.4f) weight=%.4f  npw=%d/%d\n",
                    ik, kpoints[ik][0], kpoints[ik][1], kpoints[ik][2],
                    kweights[ik], active, num_pw);
    }

    // 5. Initialize density
    RVec density_r = initial_density(basis, fft_grid);

    // 6. SCF loop
    double prev_energy = 0.0;
    SCFResult result;

    // Retain converged data for post-SCF force calculation
    std::vector<std::vector<CVec>> converged_wavefunctions;
    std::vector<std::vector<double>> converged_occupations;
    CVec converged_density_g;
    std::vector<complex_t> converged_density_g_full;
    std::vector<complex_t> converged_veff_r;

    for (int step = 1; step <= conv_params_.max_scf_steps; ++step) {
        KRONOS_TIMER("scf_step");
        auto step_start = std::chrono::high_resolution_clock::now();

        // a. Compute density in G-space via FFT
        std::vector<complex_t> density_c(num_grid);
        for (int i = 0; i < num_grid; ++i) {
            density_c[i] = complex_t{density_r[i], 0.0};
        }
        std::vector<complex_t> density_g_full(num_grid);
        fft_grid.forward(density_c, density_g_full);

        // Gather to PW coefficients
        CVec density_g(num_pw);
        fft_grid.gather_from_grid(basis, density_g_full, density_g);

        // b. Compute Hartree potential on the full FFT grid.
        //    V_H(G) = 8π n(G) / G²  (Rydberg units).
        //    Using density_g_full (FFT convention), so V_H is also in FFT
        //    convention: IFFT(V_H_FFT) = V_H(r).
        //    Only include G² ≤ ecutrho to avoid aliasing.
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
        if (xc.is_gga()) {
            RVec sigma = compute_sigma(density_g, basis, fft_grid);
            xc_result = xc.evaluate_gga(density_r, sigma, volume);
            // Compute GGA potential correction: V_gga = -2 * div(vsigma * nabla n)
            RVec vgga = compute_gga_potential(density_g, xc_result.vsigma, basis, fft_grid);
            // Add GGA correction to V_xc
            for (int i = 0; i < num_grid; ++i) {
                xc_result.vxc[i] += vgga[i];
            }
        } else {
            xc_result = xc.evaluate(density_r, volume);
        }

        // d. Build V_eff(r) = V_H(r) + V_xc(r) + V_loc(r)
        //    V_H and V_loc are evaluated on the full density grid (G² ≤
        //    ecutrho), not just the wavefunction PW basis.  V_loc is in
        //    physics convention (/Ω) and needs a factor of N_grid to
        //    match the FFT convention of V_H.
        std::vector<complex_t> veff_g(num_grid);
        for (int i = 0; i < num_grid; ++i) {
            veff_g[i] = vh_full_g[i] + ng * vloc_full_g[i];
        }

        // Inverse FFT to get V_H + V_loc in real space
        std::vector<complex_t> veff_r(num_grid);
        fft_grid.inverse(veff_g, veff_r);

        // Add V_xc in real space
        for (int i = 0; i < num_grid; ++i) {
            veff_r[i] += xc_result.vxc[i];
        }

        ham.update_veff(veff_r);

        // e. Solve eigenvalue problem at each k-point
        std::vector<std::vector<double>> all_eigenvalues;
        std::vector<std::vector<CVec>> all_wavefunctions;

        for (size_t ik = 0; ik < kpoints.size(); ++ik) {
            KRONOS_TIMER("eigensolver");
            auto h_apply = ham.get_apply_function(kpoints[ik]);
            auto precond = ham.kinetic_diagonal(kpoints[ik]);

            EigenResult eigen = eigensolver.solve(h_apply, precond, num_bands, num_pw);
            all_eigenvalues.push_back(eigen.eigenvalues);
            all_wavefunctions.push_back(std::move(eigen.eigenvectors));
        }

        // f. Find Fermi level and occupations
        FermiResult fermi = FermiSolver::find_fermi_level(
            all_eigenvalues, kweights, target_electrons,
            calc_params_.smearing, calc_params_.degauss, spin_factor);

        // g. Compute new density from wavefunctions
        //    n_out(r) = sum_nk w_k * f_nk * |psi_nk(r)|^2
        //    Mask wavefunctions to per-k active PW set (|k+G|² ≤ ecutwfc)
        //    to match QE's strictly per-k basis and eliminate noise from
        //    inactive components of the shared expanded basis.
        RVec density_out(num_grid, 0.0);
        for (size_t ik = 0; ik < kpoints.size(); ++ik) {
            auto ke_k = basis.kinetic_energies(kpoints[ik]);

            for (int n = 0; n < num_bands; ++n) {
                // fermi.occupations already includes spin_factor
                double occ = kweights[ik] * fermi.occupations[ik][n];
                if (occ < 1e-12) continue;

                // Mask to per-k active set before FFT
                CVec psi_masked(num_pw);
                for (int ig = 0; ig < num_pw; ++ig) {
                    psi_masked[ig] = (ke_k[ig] <= calc_params_.ecutwfc + 1.0e-6)
                                   ? all_wavefunctions[ik][n][ig]
                                   : complex_t{0.0, 0.0};
                }

                // Transform psi_G -> psi(r)
                std::vector<complex_t> psi_grid(num_grid, complex_t{0.0, 0.0});
                fft_grid.scatter_to_grid(basis, psi_masked, psi_grid);
                std::vector<complex_t> psi_r(num_grid);
                fft_grid.inverse(psi_grid, psi_r);

                // Accumulate |psi(r)|^2 weighted by occupation
                for (int i = 0; i < num_grid; ++i) {
                    density_out[i] += occ * std::norm(psi_r[i]);
                }
            }
        }

        // Normalize: integral n(r) dr = N_electrons
        // n(r) on grid: sum_i n(r_i) * (Omega / N_grid) = N_electrons
        double dn_sum = 0.0;
        for (int i = 0; i < num_grid; ++i) {
            dn_sum += density_out[i];
        }
        if (dn_sum > 1e-15) {
            double scale = target_electrons / (dn_sum * volume / num_grid);
            for (int i = 0; i < num_grid; ++i) {
                density_out[i] *= scale;
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

        double e_xc = xc_result.energy;

        double e_local = 0.0;
        for (int i = 0; i < num_grid; ++i) {
            e_local += std::real(std::conj(vloc_full_g[i]) * density_g_full[i]);
        }
        e_local *= volume / ng;
        double e_band = compute_band_energy(all_eigenvalues, fermi.occupations, kweights);

        // Total energy = E_band - E_H + E_xc - integral(V_xc * n)
        // (double counting correction)
        double vxc_integral = 0.0;
        for (int i = 0; i < num_grid; ++i) {
            vxc_integral += xc_result.vxc[i] * density_r[i];
        }
        vxc_integral *= volume / num_grid;

        double total_e = e_band - e_hartree + e_xc - vxc_integral;

        // Smearing entropy correction: -TS
        // For Gaussian smearing: -TS = -degauss/sqrt(pi) * sum w_k * spin * exp(-x^2)
        // For Fermi-Dirac:       -TS = -kT * sum w_k * spin * [f*ln(f) + (1-f)*ln(1-f)]
        double e_smearing = 0.0;
        if (calc_params_.smearing == SmearingType::Gaussian && calc_params_.degauss > 0) {
            const double deg = calc_params_.degauss;
            const double ef = fermi.fermi_energy;
            for (size_t ik = 0; ik < kpoints.size(); ++ik) {
                for (int n = 0; n < num_bands; ++n) {
                    double x = (all_eigenvalues[ik][n] - ef) / deg;
                    e_smearing += spin_factor * kweights[ik]
                                  * std::exp(-x * x);
                }
            }
            e_smearing *= -deg / std::sqrt(constants::pi);
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

        // Compute kinetic and nonlocal energies from band decomposition
        // E_band = E_kinetic + E_local + E_nonlocal + E_hartree_double
        // For now, approximate: E_kinetic = E_band - E_hartree - E_local - E_xc
        double e_nonlocal = 0.0;
        for (size_t ik = 0; ik < kpoints.size(); ++ik) {
            for (int n = 0; n < num_bands; ++n) {
                // fermi.occupations already includes spin_factor;
                // pass occupation=1.0 to energy() to avoid double-counting
                double occ = kweights[ik] * fermi.occupations[ik][n];
                if (occ < 1e-12) continue;
                double enl = nonlocal_pp.energy(
                    {all_wavefunctions[ik][n]}, {1.0}, kpoints[ik]);
                e_nonlocal += occ * enl;
            }
        }
        // E_band = T + 2E_H + E_loc + E_NL + ∫V_xc·n
        // => T = E_band - 2E_H - E_loc - E_NL - ∫V_xc·n
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
        result.eigenvalues = all_eigenvalues;
        result.fermi_energy_ev = fermi.fermi_energy * constants::rydberg_to_ev;

        // Retain data for post-SCF force calculation (overwritten each step;
        // the final iteration's values are the converged ones)
        converged_wavefunctions = all_wavefunctions;
        converged_occupations = fermi.occupations;
        converged_density_g = density_g;
        converged_density_g_full = density_g_full;
        converged_veff_r = veff_r;

        auto step_end = std::chrono::high_resolution_clock::now();
        double wall = std::chrono::duration<double>(step_end - step_start).count();
        print_scf_step(step, total_e, de, dn, wall);

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

        // i. Mix densities (with optional Kerker preconditioning for metals)
        if (use_kerker) {
            // Apply Kerker to residual in G-space before mixing:
            // n_out_precond = n_in + Kerker(n_out - n_in)
            RVec residual_r(num_grid);
            for (int i = 0; i < num_grid; ++i) {
                residual_r[i] = density_out[i] - density_r[i];
            }
            // FFT residual to G-space
            std::vector<complex_t> res_c(num_grid);
            for (int i = 0; i < num_grid; ++i) {
                res_c[i] = complex_t{residual_r[i], 0.0};
            }
            std::vector<complex_t> res_g_full(num_grid);
            fft_grid.forward(res_c, res_g_full);
            CVec res_g_pw(num_pw);
            fft_grid.gather_from_grid(basis, res_g_full, res_g_pw);

            // Apply Kerker filter
            CVec res_kerker = kerker.apply(res_g_pw, g_norm2_pw);

            // IFFT back to real space
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
        {
            double neg_charge = 0.0;
            for (int i = 0; i < num_grid; ++i) {
                if (density_r[i] < 0.0) {
                    neg_charge += -density_r[i];
                    density_r[i] = 0.0;
                }
            }
            // Renormalize to conserve total electron count
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
                              || calc_params_.type == CalculationType::Relax);

    if (compute_forces && !converged_density_g_full.empty()) {
        KRONOS_TIMER("forces");

        // Local pseudopotential forces (must use full density grid to match energy)
        result.local_forces = ForceCalculator::compute_local_forces(
            crystal_, pseudopotentials_, converged_density_g_full,
            grid_gcart, grid_g2, ecutrho, volume, num_grid);

        // Nonlocal pseudopotential forces
        result.nonlocal_forces = ForceCalculator::compute_nonlocal_forces(
            crystal_, basis, pseudopotentials_,
            converged_wavefunctions, converged_occupations,
            kpoints, kweights, spin_factor);

        // Total forces: Ewald + local + nonlocal
        result.forces = ForceCalculator::compute_total_forces(
            result.ewald_forces, result.local_forces, result.nonlocal_forces);

#ifdef KRONOS_HAS_SPGLIB
        // Symmetrize forces using crystal point group operations.
        // Raw forces from IBZ k-points don't have full crystal symmetry
        // because different k-points have different active PW counts.
        auto forces_unsym = result.forces;
        result.forces = symmetrize_forces(crystal_, result.forces);
#endif

        // Print force summary
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

    // Store the converged effective potential for band structure calculations
    result.converged_veff_r = std::move(converged_veff_r);

    result.total_energy_ev = result.total_energy_ry * constants::rydberg_to_ev;
    result.timing = TimerRegistry::instance().as_map();

    if (result.converged) {
        std::printf("\nCONVERGED in %d steps. Total energy: %.6f Ry (including Ewald: %.6f Ry)\n",
                    result.scf_steps, result.total_energy_ry, result.ewald_energy);
    } else {
        std::printf("\nNOT CONVERGED after %d steps. Total energy: %.6f Ry\n",
                    result.scf_steps, result.total_energy_ry);
    }

    return result;
}

} // namespace kronos
