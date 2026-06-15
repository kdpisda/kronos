#include "io/input_parser.hpp"
#include "io/upf_parser.hpp"
#include "io/output_writer.hpp"
#include "solver/scf.hpp"
#include "solver/bfgs.hpp"
#include "solver/vc_relax.hpp"
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
#include "utils/mpi_wrapper.hpp"
#include "gpu/gpu_context.hpp"
#include <iostream>
#include <map>
#include <string>

int main(int argc, char* argv[]) {
    using namespace kronos;

    // Initialize MPI (no-op in serial builds)
    mpi::init(&argc, &argv);
    const int my_rank = mpi::rank();
    const int num_procs = mpi::size();

    // Set MPI rank for structured logging
    Logger::instance().set_mpi_rank(my_rank);

    // 1. Parse command line (just input file path for v0.1)
    if (argc < 2) {
        if (my_rank == 0) {
            std::cerr << "Usage: kronos <input.yaml>" << std::endl;
        }
        mpi::finalize();
        return 1;
    }
    std::string input_file = argv[1];

    // 2. Print banner (rank 0 only)
    if (my_rank == 0) {
        std::cout << "KRONOS v0.1.0 - Kohn-Residual Optimized Numerics Over Silicon" << std::endl;
        std::cout << "================================================================" << std::endl;
        if (num_procs > 1) {
            std::cout << "Running on " << num_procs << " MPI processes" << std::endl;
        }
    }

    int exit_code = 0;

    try {
        KRONOS_TIMER("total");

        // 3. Parse input
        auto [crystal, input] = parse_input(input_file);
        Logger::instance().info("init", "Input parsed successfully");

#ifdef KRONOS_GPU_METAL
        if (input.hardware.apple_fast_mode) {
            gpu::GPUContext::instance().set_apple_fast_mode(true);
            Logger::instance().warning("apple_fast_mode",
                "fp32 GPU path active — results are not validation-grade");
        }
#endif

        // 4. Load pseudopotentials
        std::map<std::string, PseudoPotential> pseudopotentials;
        for (const auto& [symbol, path] : input.pseudopotentials) {
            pseudopotentials[symbol] = parse_upf(path);
            validate_pseudopotential(pseudopotentials[symbol]);
            if (my_rank == 0) {
                std::cout << "  Loaded PP: " << symbol << " (Z_val="
                          << pseudopotentials[symbol].z_valence << ")" << std::endl;
            }
        }

        // 5. Print calculation summary (rank 0 only)
        if (my_rank == 0) {
            std::cout << "\nCrystal: " << crystal.num_atoms() << " atoms, V = "
                      << crystal.volume() << " bohr^3" << std::endl;
            std::cout << "Ecutwfc: " << input.calculation.ecutwfc << " Ry" << std::endl;
            std::cout << "XC: " << input.calculation.xc_functional << std::endl;
        }

        // Helper: print timing summary (MPI-aware)
        auto print_timing = [&]() {
            if (num_procs > 1) {
                TimerRegistry::instance().print_summary_mpi();
            } else if (my_rank == 0) {
                TimerRegistry::instance().print_summary();
            }
        };

        // 6. Run calculation
        if (input.calculation.type == CalculationType::Relax) {
            // Geometry optimization via BFGS
            BFGSOptimizer::Params bfgs_params;
            bfgs_params.force_threshold = input.convergence.force_threshold;
            BFGSOptimizer optimizer(bfgs_params);

            auto relax_result = optimizer.optimize(
                crystal, input.calculation, input.convergence, pseudopotentials);

            // Write output with final SCF result and relaxed crystal (rank 0 only)
            if (my_rank == 0) {
                std::string output_file = "kronos.out.json";
                OutputWriter::write_json(output_file, relax_result.final_scf,
                                         relax_result.final_crystal, "relax");
                std::cout << "\nOutput written to " << output_file << std::endl;
            }
            print_timing();
            exit_code = relax_result.converged ? 0 : 1;

        } else if (input.calculation.type == CalculationType::VCRelax) {
            // Variable-cell relaxation via generalized BFGS
            VCRelaxOptimizer::Params vc_params;
            vc_params.force_threshold = input.convergence.force_threshold;
            vc_params.stress_threshold = input.convergence.stress_threshold;
            vc_params.press_target = input.calculation.press_target;
            vc_params.cell_factor = input.calculation.cell_factor;
            VCRelaxOptimizer optimizer(vc_params);

            auto vc_result = optimizer.optimize(
                crystal, input.calculation, input.convergence, pseudopotentials);

            if (my_rank == 0) {
                std::string output_file = "kronos.out.json";
                OutputWriter::write_json(output_file, vc_result.final_scf,
                                         vc_result.final_crystal, "vc-relax");
                std::printf("\nFinal cell parameters:\n");
                Mat3 final_lat = vc_result.final_crystal.lattice();
                for (int i = 0; i < 3; ++i) {
                    std::printf("  a%d = [%12.6f, %12.6f, %12.6f]\n",
                                i + 1, final_lat[i][0], final_lat[i][1], final_lat[i][2]);
                }
                std::printf("  Volume = %.4f bohr^3\n", vc_result.final_crystal.volume());
                std::printf("  Pressure = %.4f GPa\n", vc_result.final_pressure_gpa);
            }

            exit_code = vc_result.converged ? 0 : 1;

        } else if (input.calculation.type == CalculationType::Bands) {
            // Band structure calculation:
            //   1. Run SCF to convergence
            //   2. Use converged potential to compute bands along k-path

            // Step 1: SCF
            CalculationParams scf_params = input.calculation;
            scf_params.type = CalculationType::SCF;
            SCFSolver scf(crystal, scf_params, input.convergence, pseudopotentials);
            SCFResult scf_result = scf.solve();

            if (!scf_result.converged && my_rank == 0) {
                std::cerr << "Warning: SCF did not converge before band structure" << std::endl;
            }

            if (my_rank == 0) {
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

                print_timing();
            }
            exit_code = scf_result.converged ? 0 : 1;

        } else if (input.calculation.type == CalculationType::DOS) {
            // Density of states calculation:
            //   1. Run SCF to convergence
            //   2. Compute DOS from the eigenvalues

            // Step 1: SCF
            CalculationParams scf_params = input.calculation;
            scf_params.type = CalculationType::SCF;
            SCFSolver scf(crystal, scf_params, input.convergence, pseudopotentials);
            SCFResult scf_result = scf.solve();

            if (!scf_result.converged && my_rank == 0) {
                std::cerr << "Warning: SCF did not converge before DOS calculation" << std::endl;
            }

            if (my_rank == 0) {
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

                print_timing();
            }
            exit_code = scf_result.converged ? 0 : 1;

        } else {
            // Standard SCF calculation
            SCFSolver scf(crystal, input.calculation, input.convergence, pseudopotentials);
            SCFResult result = scf.solve();

            // 7. Write output (rank 0 only)
            if (my_rank == 0) {
                std::string output_file = "kronos.out.json";
                OutputWriter::write_json(output_file, result, crystal, "scf");
                std::cout << "\nOutput written to " << output_file << std::endl;
            }

            // 8. Print timing summary (MPI-aware)
            print_timing();

            exit_code = result.converged ? 0 : 1;
        }

    } catch (const InputValidationError& e) {
        if (my_rank == 0) {
            std::cerr << "Input error: " << e.what() << std::endl;
        }
        exit_code = 2;
    } catch (const UPFParseError& e) {
        if (my_rank == 0) {
            std::cerr << "Pseudopotential error: " << e.what() << std::endl;
        }
        exit_code = 3;
    } catch (const std::exception& e) {
        if (my_rank == 0) {
            std::cerr << "Error: " << e.what() << std::endl;
        }
        exit_code = 4;
    }

    mpi::finalize();
    return exit_code;
}
