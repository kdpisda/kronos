#include "io/input_parser.hpp"
#include "io/upf_parser.hpp"
#include "io/output_writer.hpp"
#include "solver/scf.hpp"
#include "solver/bfgs.hpp"
#include "postprocessing/band_structure.hpp"
#include "postprocessing/dos.hpp"
#include "hamiltonian/hamiltonian.hpp"
#include "basis/plane_wave.hpp"
#include "basis/fft_grid.hpp"
#include "potential/local_pp.hpp"
#include "potential/nonlocal_pp.hpp"
#include "potential/hartree.hpp"
#include "potential/xc.hpp"
#include "potential/gradient.hpp"
#include "solver/fermi.hpp"
#include "core/constants.hpp"
#include "utils/logger.hpp"
#include "utils/timer.hpp"
#include <iostream>
#include <map>
#include <string>

int main(int argc, char* argv[]) {
    using namespace kronos;

    // 1. Parse command line (just input file path for v0.1)
    if (argc < 2) {
        std::cerr << "Usage: kronos <input.yaml>" << std::endl;
        return 1;
    }
    std::string input_file = argv[1];

    // 2. Print banner
    std::cout << "KRONOS v0.1.0 - Kohn-Residual Optimized Numerics Over Silicon" << std::endl;
    std::cout << "================================================================" << std::endl;

    try {
        KRONOS_TIMER("total");

        // 3. Parse input
        auto [crystal, input] = parse_input(input_file);
        Logger::instance().info("init", "Input parsed successfully");

        // 4. Load pseudopotentials
        std::map<std::string, PseudoPotential> pseudopotentials;
        for (const auto& [symbol, path] : input.pseudopotentials) {
            pseudopotentials[symbol] = parse_upf(path);
            validate_pseudopotential(pseudopotentials[symbol]);
            std::cout << "  Loaded PP: " << symbol << " (Z_val="
                      << pseudopotentials[symbol].z_valence << ")" << std::endl;
        }

        // 5. Print calculation summary
        std::cout << "\nCrystal: " << crystal.num_atoms() << " atoms, V = "
                  << crystal.volume() << " bohr^3" << std::endl;
        std::cout << "Ecutwfc: " << input.calculation.ecutwfc << " Ry" << std::endl;
        std::cout << "XC: " << input.calculation.xc_functional << std::endl;

        // 6. Run calculation
        if (input.calculation.type == CalculationType::Relax) {
            // Geometry optimization via BFGS
            BFGSOptimizer::Params bfgs_params;
            bfgs_params.force_threshold = input.convergence.force_threshold;
            BFGSOptimizer optimizer(bfgs_params);

            auto relax_result = optimizer.optimize(
                crystal, input.calculation, input.convergence, pseudopotentials);

            // Write output with final SCF result and relaxed crystal
            std::string output_file = "kronos.out.json";
            OutputWriter::write_json(output_file, relax_result.final_scf,
                                     relax_result.final_crystal, "relax");
            std::cout << "\nOutput written to " << output_file << std::endl;

            TimerRegistry::instance().print_summary();
            return relax_result.converged ? 0 : 1;

        } else if (input.calculation.type == CalculationType::Bands) {
            // Band structure calculation:
            //   1. Run SCF to convergence
            //   2. Use converged potential to compute bands along k-path

            // Step 1: SCF
            CalculationParams scf_params = input.calculation;
            scf_params.type = CalculationType::SCF;
            SCFSolver scf(crystal, scf_params, input.convergence, pseudopotentials);
            SCFResult scf_result = scf.solve();

            if (!scf_result.converged) {
                std::cerr << "Warning: SCF did not converge before band structure" << std::endl;
            }

            // Write SCF output
            OutputWriter::write_json("kronos.out.json", scf_result, crystal, "bands");

            // Step 2: Use converged potential from SCF to compute bands.
            // Re-create basis, FFT grid, and Hamiltonian with the same
            // grid dimensions as the SCF (using ecutrho for density cutoff).
            PlaneWaveBasis basis(crystal, input.calculation.ecutwfc);
            double ecutrho = input.calculation.ecutrho > 0
                ? input.calculation.ecutrho : 4.0 * input.calculation.ecutwfc;
            FFTGrid fft_grid(basis, ecutrho);
            NonlocalPP nonlocal_pp(crystal, basis, pseudopotentials);
            Hamiltonian ham(crystal, basis, fft_grid, nonlocal_pp);

            // Use the converged V_eff exported from the SCF solver.
            ham.update_veff(scf_result.converged_veff_r);

            // Generate k-path (use FCC default for now; future: configurable)
            auto path_spec = BandStructureCalculator::default_path_fcc();
            auto kpath = BandStructureCalculator::generate_kpath(crystal, path_spec);

            // Compute number of bands
            int num_bands = 8;
            {
                double total_valence = 0.0;
                for (const auto& atom : crystal.atoms()) {
                    auto it = pseudopotentials.find(atom.symbol);
                    if (it != pseudopotentials.end()) {
                        total_valence += it->second.z_valence;
                    }
                }
                int num_occupied = static_cast<int>(std::ceil(total_valence / 2.0));
                num_bands = std::max(num_occupied + 4, 8);
            }

            int npw = static_cast<int>(basis.num_pw());

            auto h_apply_factory = [&](const Vec3& k_frac) {
                return ham.get_apply_function(k_frac);
            };

            auto precond_factory = [&](const Vec3& k_frac) {
                return ham.kinetic_diagonal(k_frac);
            };

            auto num_pw_func = [&](const Vec3& /*k_frac*/) {
                return npw;
            };

            std::cout << "\nComputing band structure along k-path ("
                      << kpath.kpoints.size() << " k-points)..." << std::endl;

            BandStructureCalculator::compute_bands(
                kpath, h_apply_factory, precond_factory, num_bands, num_pw_func);

            BandStructureCalculator::write_bands_gnuplot("kronos.bands.dat", kpath);
            std::cout << "Band structure written to kronos.bands.dat" << std::endl;

            TimerRegistry::instance().print_summary();
            return scf_result.converged ? 0 : 1;

        } else if (input.calculation.type == CalculationType::DOS) {
            // Density of states calculation:
            //   1. Run SCF to convergence
            //   2. Compute DOS from the eigenvalues

            // Step 1: SCF
            CalculationParams scf_params = input.calculation;
            scf_params.type = CalculationType::SCF;
            SCFSolver scf(crystal, scf_params, input.convergence, pseudopotentials);
            SCFResult scf_result = scf.solve();

            if (!scf_result.converged) {
                std::cerr << "Warning: SCF did not converge before DOS calculation" << std::endl;
            }

            // Write SCF output
            OutputWriter::write_json("kronos.out.json", scf_result, crystal, "dos");

            // Step 2: Compute DOS from SCF eigenvalues.
            // For a single k-point (Gamma), weight = 1.0.
            std::vector<double> kweights = {1.0};

            // Convert degauss from Ry to eV for the DOS calculator.
            double degauss_ev = input.calculation.degauss * constants::rydberg_to_ev;
            if (degauss_ev < 1e-6) degauss_ev = 0.05;  // sensible default

            // Determine energy range from eigenvalues.
            double e_min_ev =  1e30;
            double e_max_ev = -1e30;
            for (const auto& ek : scf_result.eigenvalues) {
                for (double e : ek) {
                    double e_ev = e * constants::rydberg_to_ev;
                    e_min_ev = std::min(e_min_ev, e_ev);
                    e_max_ev = std::max(e_max_ev, e_ev);
                }
            }
            // Expand range by 5 eV on each side.
            e_min_ev -= 5.0;
            e_max_ev += 5.0;

            int spin_factor = input.calculation.spin_polarized ? 1 : 2;

            std::cout << "\nComputing DOS (degauss = " << degauss_ev << " eV)..." << std::endl;

            auto dos_data = DOSCalculator::compute_dos(
                scf_result.eigenvalues, kweights,
                input.calculation.smearing, degauss_ev,
                e_min_ev, e_max_ev, 2001, spin_factor);

            DOSCalculator::write_dos("kronos.dos.dat", dos_data);
            std::cout << "DOS written to kronos.dos.dat" << std::endl;

            TimerRegistry::instance().print_summary();
            return scf_result.converged ? 0 : 1;

        } else {
            // Standard SCF calculation
            SCFSolver scf(crystal, input.calculation, input.convergence, pseudopotentials);
            SCFResult result = scf.solve();

            // 7. Write output
            std::string output_file = "kronos.out.json";
            OutputWriter::write_json(output_file, result, crystal, "scf");
            std::cout << "\nOutput written to " << output_file << std::endl;

            // 8. Print timing summary
            TimerRegistry::instance().print_summary();

            return result.converged ? 0 : 1;
        }

    } catch (const InputValidationError& e) {
        std::cerr << "Input error: " << e.what() << std::endl;
        return 2;
    } catch (const UPFParseError& e) {
        std::cerr << "Pseudopotential error: " << e.what() << std::endl;
        return 3;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 4;
    }
}
