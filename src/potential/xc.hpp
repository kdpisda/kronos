#pragma once

#include "core/types.hpp"

#include <memory>
#include <string>
#include <vector>

namespace kronos {

/// Result of an exchange-correlation evaluation on a real-space grid.
struct XCResult {
    RVec   exc;    ///< Energy density per grid point (Ry).
    RVec   vxc;    ///< Potential V_xc(r) per grid point (Ry).
    RVec   vsigma; ///< dE/d(sigma) per grid point (Ry) -- for GGA only.
    double energy; ///< Total XC energy  E_xc = (Omega/N) * sum_i exc[i]*n[i]  (Ry).
};

/// Exchange-correlation potential evaluator.
///
/// Wraps libxc (when available) or falls back to a built-in LDA
/// Perdew-Zunger parameterisation.
///
/// Supported functional names:
///   - "LDA_PZ"   Slater exchange + Perdew-Zunger correlation
///   - "LDA_PW"   Slater exchange + Perdew-Wang correlation
///   - "PBE"      PBE exchange + PBE correlation (GGA)
///   - "PBEsol"   PBEsol exchange + PBEsol correlation (GGA)
///
/// All energies and potentials are in Rydberg atomic units.
class XCEvaluator {
public:
    /// Construct from a functional name (case-sensitive).
    explicit XCEvaluator(const std::string& functional_name);
    ~XCEvaluator();

    // Non-copyable (owns libxc state)
    XCEvaluator(const XCEvaluator&) = delete;
    XCEvaluator& operator=(const XCEvaluator&) = delete;

    /// Evaluate the XC functional on a real-space density grid (unpolarized).
    ///
    /// @param density_r    Electron density n(r) on the real-space grid.
    /// @param cell_volume  Unit cell volume in bohr^3.
    /// @return XCResult containing exc, vxc, and total energy.
    [[nodiscard]] XCResult evaluate(const RVec& density_r,
                                    double cell_volume) const;

    /// Evaluate the spin-polarized XC functional.
    ///
    /// @param density_up   Spin-up electron density n_up(r).
    /// @param density_dn   Spin-down electron density n_dn(r).
    /// @param cell_volume  Unit cell volume in bohr^3.
    /// @return XCResult where vxc contains V_xc_up and vxc_dn (interleaved or separate).
    struct SpinXCResult {
        RVec   vxc_up;   ///< V_xc for spin-up
        RVec   vxc_dn;   ///< V_xc for spin-down
        double energy;   ///< Total E_xc
    };
    [[nodiscard]] SpinXCResult evaluate_spin(const RVec& density_up,
                                              const RVec& density_dn,
                                              double cell_volume) const;

    /// Evaluate the GGA XC functional on a real-space density grid.
    ///
    /// @param density_r    Electron density n(r) on the real-space grid.
    /// @param sigma_r      |nabla n(r)|^2 (gradient squared) on real-space grid.
    /// @param cell_volume  Unit cell volume in bohr^3.
    /// @return XCResult containing exc, vxc (= vrho), vsigma, and total energy.
    [[nodiscard]] XCResult evaluate_gga(const RVec& density_r,
                                        const RVec& sigma_r,
                                        double cell_volume) const;

    /// True if the functional is of GGA type (requires density gradients).
    [[nodiscard]] bool is_gga() const;

    /// Return the functional name supplied at construction.
    [[nodiscard]] const std::string& name() const;

private:
    std::string name_;
    bool is_gga_{false};

    // libxc functional IDs  (-1 = unused)
    int xc_func_id_{-1};   // combined XC (unused for now)
    int x_func_id_{-1};    // exchange part
    int c_func_id_{-1};    // correlation part
    bool use_combined_{false};

    // Opaque libxc handles (void* avoids <xc.h> in the public header)
    void* xc_func_{nullptr};
    void* x_func_{nullptr};
    void* c_func_{nullptr};

    void init_functional();
    void cleanup();

    // Built-in fallback for LDA_PZ when libxc is not available
    XCResult evaluate_builtin_lda_pz(const RVec& density_r,
                                     double cell_volume) const;

    // Built-in spin-polarized LDA_PZ (LSDA)
    SpinXCResult evaluate_builtin_lsda_pz(const RVec& density_up,
                                           const RVec& density_dn,
                                           double cell_volume) const;
};

} // namespace kronos
