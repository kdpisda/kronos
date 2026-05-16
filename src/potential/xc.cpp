#include "potential/xc.hpp"
#include "core/constants.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <stdexcept>
#include <iostream>

// ---------------------------------------------------------------------------
// libxc integration  (only when KRONOS_HAS_LIBXC is defined at build time)
// ---------------------------------------------------------------------------
#ifdef KRONOS_HAS_LIBXC
#include <xc.h>
#include <xc_funcs.h>
#endif

namespace kronos {

// ===================================================================
// Built-in LDA Perdew-Zunger implementation (always available)
// ===================================================================
namespace builtin {

// Slater exchange in Rydberg units.
//   ex(n) = -2 * (3/(4*pi))^{1/3} * n^{1/3}          [Ry]
//   vx(n) = (4/3) * ex(n)
static void lda_x(const double* rho, double* ex, double* vx, int np)
{
    // (3/4)*(3/pi)^{1/3}  — correct Slater exchange coefficient
    static const double ax = 0.75 * std::cbrt(3.0 / constants::pi);
    // Prefactor for energy density: -2 * ax  (Ry units; the factor of 2
    // comes from converting Hartree -> Ry:  ex_Ry = 2 * ex_Ha).
    static const double fx = -2.0 * ax;

    for (int i = 0; i < np; ++i) {
        const double n = std::max(rho[i], 0.0);
        if (n < 1.0e-30) {
            ex[i] = 0.0;
            vx[i] = 0.0;
        } else {
            const double n13 = std::cbrt(n);
            ex[i] = fx * n13;
            vx[i] = (4.0 / 3.0) * fx * n13;
        }
    }
}

// Perdew-Zunger parameterisation of Ceperley-Alder correlation (Rydberg).
//
// For rs >= 1:  ec = gamma / (1 + beta1*sqrt(rs) + beta2*rs)
// For rs <  1:  ec = A*ln(rs) + B + C*rs*ln(rs) + D*rs
//
// PZ paper (1981) Table IV gives Hartree parameters.  Convert to Rydberg
// by multiplying energies by 2 (beta1, beta2 are dimensionless).
//
// Hartree:  gamma = -0.1423,  A = 0.0311, B = -0.048, C = 0.002, D = -0.0116
// Rydberg:  gamma = -0.2846,  A = 0.0622, B = -0.096, C = 0.004, D = -0.0232
static void lda_c_pz(const double* rho, double* ec, double* vc, int np)
{
    // PZ parameters in Rydberg units (unpolarised)
    // rs >= 1
    constexpr double gamma = -0.2846;
    constexpr double beta1 = 1.0529;
    constexpr double beta2 = 0.3334;
    // rs < 1
    constexpr double A =  0.0622;
    constexpr double B = -0.0960;
    constexpr double C =  0.0040;
    constexpr double D = -0.0232;

    // rs = (3 / (4*pi*n))^{1/3}
    static const double rs_prefactor = std::cbrt(3.0 / (4.0 * constants::pi));

    for (int i = 0; i < np; ++i) {
        const double n = std::max(rho[i], 0.0);
        if (n < 1.0e-30) {
            ec[i] = 0.0;
            vc[i] = 0.0;
            continue;
        }

        const double rs = rs_prefactor / std::cbrt(n);

        if (rs >= 1.0) {
            const double sqrs = std::sqrt(rs);
            const double denom = 1.0 + beta1 * sqrs + beta2 * rs;
            ec[i] = gamma / denom;
            // d(ec)/d(rs) = -gamma * (beta1/(2*sqrt(rs)) + beta2) / denom^2
            const double dec_drs = -gamma * (0.5 * beta1 / sqrs + beta2)
                                   / (denom * denom);
            // vc = ec - (rs/3) * dec/drs
            vc[i] = ec[i] - (rs / 3.0) * dec_drs;
        } else {
            const double lnrs = std::log(rs);
            ec[i] = A * lnrs + B + C * rs * lnrs + D * rs;
            // dec/drs = A/rs + C*ln(rs) + C + D
            const double dec_drs = A / rs + C * lnrs + C + D;
            vc[i] = ec[i] - (rs / 3.0) * dec_drs;
        }
    }
}

} // namespace builtin

// ===================================================================
// XCEvaluator implementation
// ===================================================================

XCEvaluator::XCEvaluator(const std::string& functional_name)
    : name_(functional_name)
{
    init_functional();
}

XCEvaluator::~XCEvaluator()
{
    cleanup();
}

const std::string& XCEvaluator::name() const
{
    return name_;
}

bool XCEvaluator::is_gga() const
{
    return is_gga_;
}

bool XCEvaluator::is_hybrid() const
{
    return hybrid_type_ != HybridType::None;
}

// ---------------------------------------------------------------------------
// init_functional  --  map name -> libxc IDs, then initialise
// ---------------------------------------------------------------------------

void XCEvaluator::init_functional()
{
    // Resolve functional name to exchange / correlation IDs
    if (name_ == "LDA_PZ") {
        is_gga_ = false;
#ifdef KRONOS_HAS_LIBXC
        x_func_id_ = XC_LDA_X;
        c_func_id_ = XC_LDA_C_PZ;
#endif
    } else if (name_ == "LDA_PW") {
        is_gga_ = false;
#ifdef KRONOS_HAS_LIBXC
        x_func_id_ = XC_LDA_X;
        c_func_id_ = XC_LDA_C_PW;
#endif
    } else if (name_ == "PBE") {
        is_gga_ = true;
#ifdef KRONOS_HAS_LIBXC
        x_func_id_ = XC_GGA_X_PBE;
        c_func_id_ = XC_GGA_C_PBE;
#endif
    } else if (name_ == "PBEsol") {
        is_gga_ = true;
#ifdef KRONOS_HAS_LIBXC
        x_func_id_ = XC_GGA_X_PBE_SOL;
        c_func_id_ = XC_GGA_C_PBE_SOL;
#endif
    } else if (name_ == "PBE0") {
        is_gga_ = true;
        hybrid_type_ = HybridType::PBE0;
        exx_fraction_ = 0.25;
        screening_parameter_ = 0.0;
        // Semi-local part: (1-α) PBE exchange + PBE correlation
#ifdef KRONOS_HAS_LIBXC
        x_func_id_ = XC_GGA_X_PBE;
        c_func_id_ = XC_GGA_C_PBE;
#endif
    } else if (name_ == "HSE06") {
        is_gga_ = true;
        hybrid_type_ = HybridType::HSE06;
        exx_fraction_ = 0.25;
        screening_parameter_ = 0.11;  // bohr⁻¹
        // Semi-local part: short-range PBE exchange + PBE correlation
#ifdef KRONOS_HAS_LIBXC
        x_func_id_ = XC_GGA_X_PBE;
        c_func_id_ = XC_GGA_C_PBE;
#endif
    } else {
        throw std::invalid_argument("XCEvaluator: unknown functional '" +
                                    name_ + "'");
    }

#ifdef KRONOS_HAS_LIBXC
    // Allocate and initialise libxc functional structs
    if (x_func_id_ >= 0) {
        x_func_ = new xc_func_type;
        if (xc_func_init(static_cast<xc_func_type*>(x_func_),
                          x_func_id_, XC_UNPOLARIZED) != 0) {
            delete static_cast<xc_func_type*>(x_func_);
            x_func_ = nullptr;
            throw std::runtime_error(
                "XCEvaluator: failed to initialise libxc exchange functional");
        }
    }
    if (c_func_id_ >= 0) {
        c_func_ = new xc_func_type;
        if (xc_func_init(static_cast<xc_func_type*>(c_func_),
                          c_func_id_, XC_UNPOLARIZED) != 0) {
            delete static_cast<xc_func_type*>(c_func_);
            c_func_ = nullptr;
            throw std::runtime_error(
                "XCEvaluator: failed to initialise libxc correlation functional");
        }
    }
#endif
}

// ---------------------------------------------------------------------------
// cleanup
// ---------------------------------------------------------------------------

void XCEvaluator::cleanup()
{
#ifdef KRONOS_HAS_LIBXC
    if (x_func_) {
        xc_func_end(static_cast<xc_func_type*>(x_func_));
        delete static_cast<xc_func_type*>(x_func_);
        x_func_ = nullptr;
    }
    if (c_func_) {
        xc_func_end(static_cast<xc_func_type*>(c_func_));
        delete static_cast<xc_func_type*>(c_func_);
        c_func_ = nullptr;
    }
    if (xc_func_) {
        xc_func_end(static_cast<xc_func_type*>(xc_func_));
        delete static_cast<xc_func_type*>(xc_func_);
        xc_func_ = nullptr;
    }
#endif
}

// ---------------------------------------------------------------------------
// evaluate
// ---------------------------------------------------------------------------

XCResult XCEvaluator::evaluate(const RVec& density_r,
                               double cell_volume) const
{
#ifdef KRONOS_HAS_LIBXC
    const int np = static_cast<int>(density_r.size());

    XCResult result;
    result.exc.assign(np, 0.0);
    result.vxc.assign(np, 0.0);
    result.energy = 0.0;

    // Temporary buffers for each functional part
    RVec exc_tmp(np, 0.0);
    RVec vxc_tmp(np, 0.0);

    // Ensure the density has no negative values (libxc requirement)
    RVec rho_safe(np);
    for (int i = 0; i < np; ++i) {
        rho_safe[i] = std::max(density_r[i], 0.0);
    }

    auto eval_func = [&](void* func_ptr) {
        auto* func = static_cast<xc_func_type*>(func_ptr);
        if (!func) return;

        std::fill(exc_tmp.begin(), exc_tmp.end(), 0.0);
        std::fill(vxc_tmp.begin(), vxc_tmp.end(), 0.0);

        switch (func->info->family) {
        case XC_FAMILY_LDA:
        case XC_FAMILY_HYB_LDA:
            xc_lda_exc_vxc(func, np, rho_safe.data(),
                           exc_tmp.data(), vxc_tmp.data());
            break;

        case XC_FAMILY_GGA:
        case XC_FAMILY_HYB_GGA:
            // GGA requires density gradients |nabla n|^2.
            // For v0.1 we fall back to LDA-level evaluation with a warning.
            std::cerr << "[kronos] WARNING: GGA functional '" << name_
                      << "' requested but gradient computation is not yet "
                         "implemented; falling back to LDA-level evaluation.\n";
            // Provide zero sigma (|nabla n|^2 = 0) as a rough placeholder.
            {
                RVec sigma(np, 0.0);
                RVec vsigma(np, 0.0);
                xc_gga_exc_vxc(func, np, rho_safe.data(), sigma.data(),
                                exc_tmp.data(), vxc_tmp.data(), vsigma.data());
            }
            break;

        default:
            throw std::runtime_error(
                "XCEvaluator: unsupported libxc functional family");
        }

        // Accumulate (libxc returns energy densities per particle in Hartree;
        // convert to Ry by multiplying by 2)
        for (int i = 0; i < np; ++i) {
            result.exc[i] += 2.0 * exc_tmp[i];
            result.vxc[i] += 2.0 * vxc_tmp[i];
        }
    };

    // Evaluate exchange part
    eval_func(x_func_);
    // Scale semi-local exchange for hybrids: (1-α) factor
    if (exchange_scale_ != 1.0) {
        for (int i = 0; i < np; ++i) {
            result.exc[i] *= exchange_scale_;
            result.vxc[i] *= exchange_scale_;
        }
    }
    // Evaluate correlation part
    eval_func(c_func_);
    // If a combined XC functional was used, evaluate it too
    if (use_combined_) {
        eval_func(xc_func_);
    }

    // Total XC energy: E_xc = (Omega / N_grid) * sum_i exc[i] * n[i]
    const double dv = cell_volume / static_cast<double>(np);
    double esum = 0.0;
    for (int i = 0; i < np; ++i) {
        esum += result.exc[i] * rho_safe[i];
    }
    result.energy = dv * esum;

    return result;

#else
    // No libxc -- use built-in implementations
    if (is_gga_) {
        std::cerr << "[kronos] WARNING: GGA functional '" << name_
                  << "' requested but libxc is not available; "
                     "falling back to built-in LDA_PZ.\n";
    } else if (name_ != "LDA_PZ" && name_ != "LDA_PW") {
        std::cerr << "[kronos] WARNING: functional '" << name_
                  << "' requested but libxc is not available; "
                     "falling back to built-in LDA_PZ.\n";
    }
    return evaluate_builtin_lda_pz(density_r, cell_volume);
#endif
}

// ---------------------------------------------------------------------------
// evaluate_gga  --  GGA evaluation with density gradient sigma
// ---------------------------------------------------------------------------

XCResult XCEvaluator::evaluate_gga(const RVec& density_r,
                                    const RVec& sigma_r,
                                    double cell_volume) const
{
    const int np = static_cast<int>(density_r.size());
    assert(static_cast<int>(sigma_r.size()) == np);

#ifdef KRONOS_HAS_LIBXC
    XCResult result;
    result.exc.assign(np, 0.0);
    result.vxc.assign(np, 0.0);
    result.vsigma.assign(np, 0.0);
    result.energy = 0.0;

    // Ensure the density and sigma have no negative values (libxc requirement)
    RVec rho_safe(np);
    RVec sigma_safe(np);
    for (int i = 0; i < np; ++i) {
        rho_safe[i] = std::max(density_r[i], 0.0);
        sigma_safe[i] = std::max(sigma_r[i], 0.0);
    }

    // Temporary buffers for each functional part
    RVec exc_tmp(np, 0.0);
    RVec vrho_tmp(np, 0.0);
    RVec vsigma_tmp(np, 0.0);

    auto eval_func = [&](void* func_ptr) {
        auto* func = static_cast<xc_func_type*>(func_ptr);
        if (!func) return;

        std::fill(exc_tmp.begin(), exc_tmp.end(), 0.0);
        std::fill(vrho_tmp.begin(), vrho_tmp.end(), 0.0);
        std::fill(vsigma_tmp.begin(), vsigma_tmp.end(), 0.0);

        switch (func->info->family) {
        case XC_FAMILY_LDA:
        case XC_FAMILY_HYB_LDA:
            // LDA part of a GGA evaluation: no sigma dependency
            xc_lda_exc_vxc(func, np, rho_safe.data(),
                           exc_tmp.data(), vrho_tmp.data());
            // vsigma_tmp stays zero for LDA
            break;

        case XC_FAMILY_GGA:
        case XC_FAMILY_HYB_GGA:
            xc_gga_exc_vxc(func, np, rho_safe.data(), sigma_safe.data(),
                            exc_tmp.data(), vrho_tmp.data(), vsigma_tmp.data());
            break;

        default:
            throw std::runtime_error(
                "XCEvaluator::evaluate_gga: unsupported libxc functional family");
        }

        // Accumulate (libxc returns energy densities per particle in Hartree;
        // convert to Ry by multiplying by 2)
        for (int i = 0; i < np; ++i) {
            result.exc[i]    += 2.0 * exc_tmp[i];
            result.vxc[i]    += 2.0 * vrho_tmp[i];
            result.vsigma[i] += 2.0 * vsigma_tmp[i];
        }
    };

    // Evaluate exchange part
    eval_func(x_func_);
    // Scale semi-local exchange for hybrids: (1-α) factor
    if (exchange_scale_ != 1.0) {
        for (int i = 0; i < np; ++i) {
            result.exc[i] *= exchange_scale_;
            result.vxc[i] *= exchange_scale_;
            result.vsigma[i] *= exchange_scale_;
        }
    }
    // Evaluate correlation part
    eval_func(c_func_);
    // If a combined XC functional was used, evaluate it too
    if (use_combined_) {
        eval_func(xc_func_);
    }

    // Total XC energy: E_xc = (Omega / N_grid) * sum_i exc[i] * n[i]
    const double dv = cell_volume / static_cast<double>(np);
    double esum = 0.0;
    for (int i = 0; i < np; ++i) {
        esum += result.exc[i] * rho_safe[i];
    }
    result.energy = dv * esum;

    return result;

#else
    // No libxc -- use built-in PBE if available
    if (name_ == "PBE" || name_ == "PBEsol" || name_ == "PBE0" || name_ == "HSE06") {
        return evaluate_builtin_pbe(density_r, sigma_r, cell_volume);
    }
    std::cerr << "[kronos] WARNING: GGA functional '" << name_
              << "' requested but libxc is not available; "
                 "falling back to built-in LDA_PZ.\n";
    XCResult result = evaluate_builtin_lda_pz(density_r, cell_volume);
    result.vsigma.assign(np, 0.0);
    return result;
#endif
}

// ---------------------------------------------------------------------------
// Built-in LDA_PZ evaluation (no libxc dependency)
// ---------------------------------------------------------------------------

XCResult XCEvaluator::evaluate_builtin_lda_pz(const RVec& density_r,
                                               double cell_volume) const
{
    const int np = static_cast<int>(density_r.size());

    XCResult result;
    result.exc.resize(np);
    result.vxc.resize(np);
    result.energy = 0.0;

    // Temporary buffers for exchange and correlation parts
    RVec ex(np), vx(np);
    RVec ec(np), vc(np);

    builtin::lda_x(density_r.data(), ex.data(), vx.data(), np);
    builtin::lda_c_pz(density_r.data(), ec.data(), vc.data(), np);

    // Sum exchange (scaled) + correlation
    const double dv = cell_volume / static_cast<double>(np);
    double esum = 0.0;
    const double xs = exchange_scale_;

    for (int i = 0; i < np; ++i) {
        result.exc[i] = xs * ex[i] + ec[i];
        result.vxc[i] = xs * vx[i] + vc[i];
        esum += result.exc[i] * std::max(density_r[i], 0.0);
    }

    result.energy = dv * esum;
    return result;
}

// ---------------------------------------------------------------------------
// Spin-polarized evaluation (LSDA)
// ---------------------------------------------------------------------------

XCEvaluator::SpinXCResult XCEvaluator::evaluate_spin(
    const RVec& density_up,
    const RVec& density_dn,
    double cell_volume) const
{
#ifdef KRONOS_HAS_LIBXC
    const int np = static_cast<int>(density_up.size());
    assert(static_cast<int>(density_dn.size()) == np);

    SpinXCResult result;
    result.vxc_up.assign(np, 0.0);
    result.vxc_dn.assign(np, 0.0);
    result.energy = 0.0;

    // Pack spin densities for libxc: [n_up(0), n_dn(0), n_up(1), n_dn(1), ...]
    std::vector<double> rho_spin(2 * np);
    for (int i = 0; i < np; ++i) {
        rho_spin[2*i]     = std::max(density_up[i], 0.0);
        rho_spin[2*i + 1] = std::max(density_dn[i], 0.0);
    }

    // Initialize spin-polarized functionals
    // We need separate init for XC_POLARIZED
    auto eval_func_spin = [&](int func_id) {
        if (func_id < 0) return;

        xc_func_type func;
        if (xc_func_init(&func, func_id, XC_POLARIZED) != 0) {
            throw std::runtime_error(
                "XCEvaluator: failed to init spin-polarized libxc functional");
        }

        std::vector<double> exc_tmp(np, 0.0);
        std::vector<double> vrho_tmp(2 * np, 0.0);

        switch (func.info->family) {
        case XC_FAMILY_LDA:
        case XC_FAMILY_HYB_LDA:
            xc_lda_exc_vxc(&func, np, rho_spin.data(),
                           exc_tmp.data(), vrho_tmp.data());
            break;
        case XC_FAMILY_GGA:
        case XC_FAMILY_HYB_GGA:
            // For GGA spin-polarized, we need sigma_uu, sigma_ud, sigma_dd
            // For now, fall back to LDA-level (sigma=0)
            {
                std::vector<double> sigma(3 * np, 0.0);
                std::vector<double> vsigma(3 * np, 0.0);
                xc_gga_exc_vxc(&func, np, rho_spin.data(), sigma.data(),
                                exc_tmp.data(), vrho_tmp.data(), vsigma.data());
            }
            break;
        default:
            xc_func_end(&func);
            throw std::runtime_error(
                "XCEvaluator::evaluate_spin: unsupported functional family");
        }

        // Accumulate (convert Hartree → Ry by ×2)
        const double dv = cell_volume / static_cast<double>(np);
        for (int i = 0; i < np; ++i) {
            double n_total = rho_spin[2*i] + rho_spin[2*i + 1];
            result.energy += dv * 2.0 * exc_tmp[i] * n_total;
            result.vxc_up[i] += 2.0 * vrho_tmp[2*i];
            result.vxc_dn[i] += 2.0 * vrho_tmp[2*i + 1];
        }

        xc_func_end(&func);
    };

    eval_func_spin(x_func_id_);
    // Scale semi-local exchange for hybrids
    if (exchange_scale_ != 1.0) {
        for (int i = 0; i < np; ++i) {
            result.vxc_up[i] *= exchange_scale_;
            result.vxc_dn[i] *= exchange_scale_;
        }
        result.energy *= exchange_scale_;
    }
    eval_func_spin(c_func_id_);

    return result;

#else
    // No libxc — use built-in LSDA PZ
    if (is_gga_) {
        std::cerr << "[kronos] WARNING: GGA spin-polarized requested but libxc "
                     "not available; falling back to built-in LSDA_PZ.\n";
    }
    return evaluate_builtin_lsda_pz(density_up, density_dn, cell_volume);
#endif
}

// ---------------------------------------------------------------------------
// Built-in LSDA Perdew-Zunger (spin-polarized)
// ---------------------------------------------------------------------------

namespace builtin {

// Spin-polarized Slater exchange in Rydberg units.
// ex_up = ex(2*n_up) / 2, ex_dn = ex(2*n_dn) / 2 (scaling relation)
// vx_up = vx(2*n_up), vx_dn = vx(2*n_dn)
static void lsda_x(const double* rho_up, const double* rho_dn,
                    double* ex, double* vx_up, double* vx_dn, int np)
{
    static const double ax = 0.75 * std::cbrt(3.0 / constants::pi);
    static const double fx = -2.0 * ax;
    // Spin scaling: 2^(1/3) factor
    static const double f213 = std::cbrt(2.0);

    for (int i = 0; i < np; ++i) {
        const double nu = std::max(rho_up[i], 0.0);
        const double nd = std::max(rho_dn[i], 0.0);
        const double n = nu + nd;
        if (n < 1.0e-30) {
            ex[i] = 0.0;
            vx_up[i] = 0.0;
            vx_dn[i] = 0.0;
            continue;
        }

        // Spin-polarized exchange: e_x = (e_x(2*n_up) * n_up + e_x(2*n_dn) * n_dn) / n
        double ex_u = (nu > 1e-30) ? fx * std::cbrt(2.0 * nu) : 0.0;
        double ex_d = (nd > 1e-30) ? fx * std::cbrt(2.0 * nd) : 0.0;
        ex[i] = (ex_u * nu + ex_d * nd) / n;
        vx_up[i] = (nu > 1e-30) ? (4.0 / 3.0) * fx * std::cbrt(2.0 * nu) : 0.0;
        vx_dn[i] = (nd > 1e-30) ? (4.0 / 3.0) * fx * std::cbrt(2.0 * nd) : 0.0;
    }
}

// Spin-polarized PZ correlation using the VWN interpolation formula:
// ec(rs, zeta) = ec_P(rs) + [ec_F(rs) - ec_P(rs)] * f(zeta)
// where f(zeta) = [(1+zeta)^(4/3) + (1-zeta)^(4/3) - 2] / (2*(2^(1/3)-1))
// ec_P = paramagnetic (unpolarized), ec_F = ferromagnetic (fully polarized)
static void lsda_c_pz(const double* rho_up, const double* rho_dn,
                       double* ec, double* vc_up, double* vc_dn, int np)
{
    // PZ parameters (Rydberg): paramagnetic
    constexpr double gamma_p = -0.2846;
    constexpr double beta1_p = 1.0529;
    constexpr double beta2_p = 0.3334;
    constexpr double A_p =  0.0622;
    constexpr double B_p = -0.0960;
    constexpr double C_p =  0.0040;
    constexpr double D_p = -0.0232;

    // PZ parameters (Rydberg): ferromagnetic
    constexpr double gamma_f = -0.1686;
    constexpr double beta1_f = 1.3981;
    constexpr double beta2_f = 0.2611;
    constexpr double A_f =  0.0311;
    constexpr double B_f = -0.0538;
    constexpr double C_f =  0.0014;
    constexpr double D_f = -0.0096;

    static const double rs_prefactor = std::cbrt(3.0 / (4.0 * constants::pi));
    static const double f_denom = 2.0 * (std::cbrt(2.0) - 1.0);

    // Helper: compute ec and dec/drs for given parameters
    auto ec_pz = [](double rs, double gamma, double beta1, double beta2,
                     double A, double B, double C, double D,
                     double& e_c, double& de_drs) {
        if (rs >= 1.0) {
            double sqrs = std::sqrt(rs);
            double denom = 1.0 + beta1 * sqrs + beta2 * rs;
            e_c = gamma / denom;
            de_drs = -gamma * (0.5 * beta1 / sqrs + beta2) / (denom * denom);
        } else {
            double lnrs = std::log(rs);
            e_c = A * lnrs + B + C * rs * lnrs + D * rs;
            de_drs = A / rs + C * lnrs + C + D;
        }
    };

    for (int i = 0; i < np; ++i) {
        const double nu = std::max(rho_up[i], 0.0);
        const double nd = std::max(rho_dn[i], 0.0);
        const double n = nu + nd;
        if (n < 1.0e-30) {
            ec[i] = 0.0;
            vc_up[i] = 0.0;
            vc_dn[i] = 0.0;
            continue;
        }

        const double rs = rs_prefactor / std::cbrt(n);
        const double zeta = (nu - nd) / n;  // spin polarization [-1, 1]

        // f(zeta) spin interpolation function
        double zp1 = std::max(1.0 + zeta, 0.0);
        double zm1 = std::max(1.0 - zeta, 0.0);
        double f_zeta = (std::cbrt(zp1 * zp1 * zp1 * zp1) +
                         std::cbrt(zm1 * zm1 * zm1 * zm1) - 2.0) / f_denom;

        // df/dzeta = (4/3) * [(1+z)^(1/3) - (1-z)^(1/3)] / (2*(2^(1/3)-1))
        double df_dzeta = 0.0;
        if (std::abs(zeta) < 1.0 - 1e-12) {
            df_dzeta = (4.0 / 3.0) * (std::cbrt(zp1) - std::cbrt(zm1)) / f_denom;
        }

        double ec_p, dec_p_drs, ec_f, dec_f_drs;
        ec_pz(rs, gamma_p, beta1_p, beta2_p, A_p, B_p, C_p, D_p, ec_p, dec_p_drs);
        ec_pz(rs, gamma_f, beta1_f, beta2_f, A_f, B_f, C_f, D_f, ec_f, dec_f_drs);

        // Interpolated correlation energy
        ec[i] = ec_p + (ec_f - ec_p) * f_zeta;

        // Potential: vc = ec - (rs/3) * dec/drs + (ec_f - ec_p) * df/dzeta * dzeta/dn
        double dec_drs = dec_p_drs + (dec_f_drs - dec_p_drs) * f_zeta;
        double vc_common = ec[i] - (rs / 3.0) * dec_drs;
        double delta_ec = ec_f - ec_p;

        // dzeta/dn_up = (1-zeta)/n,  dzeta/dn_dn = -(1+zeta)/n
        vc_up[i] = vc_common + delta_ec * df_dzeta * (1.0 - zeta) / 1.0;
        vc_dn[i] = vc_common - delta_ec * df_dzeta * (1.0 + zeta) / 1.0;
        // Note: the (ec_f-ec_p)*f(zeta) contribution is already in vc_common via ec[i]
        // The extra terms come from the zeta-dependence
    }
}

} // namespace builtin (lsda)

XCEvaluator::SpinXCResult XCEvaluator::evaluate_builtin_lsda_pz(
    const RVec& density_up,
    const RVec& density_dn,
    double cell_volume) const
{
    const int np = static_cast<int>(density_up.size());

    SpinXCResult result;
    result.vxc_up.resize(np);
    result.vxc_dn.resize(np);
    result.energy = 0.0;

    RVec ex(np), vx_up(np), vx_dn(np);
    RVec ec(np), vc_up(np), vc_dn(np);

    builtin::lsda_x(density_up.data(), density_dn.data(),
                     ex.data(), vx_up.data(), vx_dn.data(), np);
    builtin::lsda_c_pz(density_up.data(), density_dn.data(),
                        ec.data(), vc_up.data(), vc_dn.data(), np);

    const double dv = cell_volume / static_cast<double>(np);
    double esum = 0.0;
    const double xs = exchange_scale_;

    for (int i = 0; i < np; ++i) {
        double exc = xs * ex[i] + ec[i];
        result.vxc_up[i] = xs * vx_up[i] + vc_up[i];
        result.vxc_dn[i] = xs * vx_dn[i] + vc_dn[i];
        double n = std::max(density_up[i], 0.0) + std::max(density_dn[i], 0.0);
        esum += exc * n;
    }

    result.energy = dv * esum;
    return result;
}

// ---------------------------------------------------------------------------
// evaluate_spin_gga  --  Spin-polarized GGA evaluation
// ---------------------------------------------------------------------------

XCEvaluator::SpinGGAResult XCEvaluator::evaluate_spin_gga(
    const RVec& density_up,
    const RVec& density_dn,
    const RVec& sigma_uu,
    const RVec& sigma_ud,
    const RVec& sigma_dd,
    double cell_volume) const
{
    const int np = static_cast<int>(density_up.size());
    assert(static_cast<int>(density_dn.size()) == np);
    assert(static_cast<int>(sigma_uu.size()) == np);
    assert(static_cast<int>(sigma_ud.size()) == np);
    assert(static_cast<int>(sigma_dd.size()) == np);

#ifdef KRONOS_HAS_LIBXC
    SpinGGAResult result;
    result.vxc_up.assign(np, 0.0);
    result.vxc_dn.assign(np, 0.0);
    result.vsigma_uu.assign(np, 0.0);
    result.vsigma_ud.assign(np, 0.0);
    result.vsigma_dd.assign(np, 0.0);
    result.energy = 0.0;

    // Pack spin densities for libxc: [n_up(0), n_dn(0), n_up(1), n_dn(1), ...]
    std::vector<double> rho_spin(2 * np);
    for (int i = 0; i < np; ++i) {
        rho_spin[2*i]     = std::max(density_up[i], 0.0);
        rho_spin[2*i + 1] = std::max(density_dn[i], 0.0);
    }

    // Pack sigma for libxc: [sigma_uu(0), sigma_ud(0), sigma_dd(0), ...]
    std::vector<double> sigma_packed(3 * np);
    for (int i = 0; i < np; ++i) {
        sigma_packed[3*i]     = std::max(sigma_uu[i], 0.0);
        sigma_packed[3*i + 1] = sigma_ud[i];  // can be negative
        sigma_packed[3*i + 2] = std::max(sigma_dd[i], 0.0);
    }

    auto eval_func_spin = [&](int func_id) {
        if (func_id < 0) return;

        xc_func_type func;
        if (xc_func_init(&func, func_id, XC_POLARIZED) != 0) {
            throw std::runtime_error(
                "XCEvaluator::evaluate_spin_gga: failed to init spin-polarized libxc functional");
        }

        std::vector<double> exc_tmp(np, 0.0);
        std::vector<double> vrho_tmp(2 * np, 0.0);
        std::vector<double> vsigma_tmp(3 * np, 0.0);

        switch (func.info->family) {
        case XC_FAMILY_LDA:
        case XC_FAMILY_HYB_LDA:
            xc_lda_exc_vxc(&func, np, rho_spin.data(),
                           exc_tmp.data(), vrho_tmp.data());
            // vsigma_tmp stays zero for LDA
            break;
        case XC_FAMILY_GGA:
        case XC_FAMILY_HYB_GGA:
            xc_gga_exc_vxc(&func, np, rho_spin.data(), sigma_packed.data(),
                           exc_tmp.data(), vrho_tmp.data(), vsigma_tmp.data());
            break;
        default:
            xc_func_end(&func);
            throw std::runtime_error(
                "XCEvaluator::evaluate_spin_gga: unsupported functional family");
        }

        // Accumulate (convert Hartree -> Ry by x2)
        const double dv = cell_volume / static_cast<double>(np);
        for (int i = 0; i < np; ++i) {
            double n_total = rho_spin[2*i] + rho_spin[2*i + 1];
            result.energy     += dv * 2.0 * exc_tmp[i] * n_total;
            result.vxc_up[i]  += 2.0 * vrho_tmp[2*i];
            result.vxc_dn[i]  += 2.0 * vrho_tmp[2*i + 1];
            result.vsigma_uu[i] += 2.0 * vsigma_tmp[3*i];
            result.vsigma_ud[i] += 2.0 * vsigma_tmp[3*i + 1];
            result.vsigma_dd[i] += 2.0 * vsigma_tmp[3*i + 2];
        }

        xc_func_end(&func);
    };

    eval_func_spin(x_func_id_);
    // Scale semi-local exchange for hybrids
    if (exchange_scale_ != 1.0) {
        for (int i = 0; i < np; ++i) {
            result.vxc_up[i] *= exchange_scale_;
            result.vxc_dn[i] *= exchange_scale_;
            result.vsigma_uu[i] *= exchange_scale_;
            result.vsigma_ud[i] *= exchange_scale_;
            result.vsigma_dd[i] *= exchange_scale_;
        }
        result.energy *= exchange_scale_;
    }
    eval_func_spin(c_func_id_);

    return result;

#else
    // No libxc -- use built-in PBE if GGA, otherwise LSDA fallback
    if (is_gga_) {
        return evaluate_builtin_spin_pbe(density_up, density_dn,
                                          sigma_uu, sigma_ud, sigma_dd,
                                          cell_volume);
    }

    // For LDA, fall back to LSDA with zero vsigma
    std::cerr << "[kronos] WARNING: spin-GGA requested for LDA functional; "
                 "using LSDA (sigma ignored).\n";
    auto lsda_result = evaluate_builtin_lsda_pz(density_up, density_dn, cell_volume);
    SpinGGAResult result;
    result.vxc_up = std::move(lsda_result.vxc_up);
    result.vxc_dn = std::move(lsda_result.vxc_dn);
    result.vsigma_uu.assign(np, 0.0);
    result.vsigma_ud.assign(np, 0.0);
    result.vsigma_dd.assign(np, 0.0);
    result.energy = lsda_result.energy;
    return result;
#endif
}

// ---------------------------------------------------------------------------
// Built-in PBE exchange-correlation implementation
// ---------------------------------------------------------------------------
//
// PBE exchange:  F_x(s) = 1 + kappa - kappa / (1 + mu*s^2/kappa)
//   where s = |nabla n| / (2 * k_F * n),  k_F = (3*pi^2*n)^{1/3}
//   mu = 0.2195149727645171,  kappa = 0.804
//
// PBE correlation:  Uses LDA correlation + gradient correction
//   H(rs, zeta, t) = gamma * ln(1 + beta/gamma * t^2 * (1 + A*t^2)/(1 + A*t^2 + A^2*t^4))
//   where t = |nabla n| / (2 * k_s * n),  k_s = sqrt(4*k_F/pi)
//   gamma = (1 - ln 2) / pi^2 ≈ 0.031091,  beta ≈ 0.066725
// ---------------------------------------------------------------------------

namespace builtin {

// PBE parameters
static constexpr double PBE_mu = 0.2195149727645171;
static constexpr double PBE_kappa = 0.804;
static constexpr double PBE_beta = 0.066725;
static constexpr double PBE_gamma = 0.031091;  // (1-ln2)/pi^2

/// Compute unpolarized PBE exchange energy density and potentials.
/// All in Rydberg units.
///
/// Returns (via pointers): exc_x, vrho_x, vsigma_x per grid point
static void pbe_x_unpol(const double* rho, const double* sigma,
                         double* exc, double* vrho, double* vsigma, int np)
{
    // Slater exchange coefficient (Ry)
    static const double ax = -2.0 * 0.75 * std::cbrt(3.0 / constants::pi);

    for (int i = 0; i < np; ++i) {
        const double n = std::max(rho[i], 1e-30);
        const double sig = std::max(sigma[i], 0.0);

        // k_F = (3*pi^2*n)^{1/3} -- note: in atomic units
        const double kf = std::cbrt(3.0 * constants::pi * constants::pi * n);
        const double n13 = std::cbrt(n);

        // s = |nabla n| / (2 * k_F * n)
        const double gn = std::sqrt(sig);
        const double s = gn / (2.0 * kf * n);
        const double s2 = s * s;

        // F_x(s) = 1 + kappa - kappa / (1 + mu*s^2/kappa)
        const double denom = 1.0 + PBE_mu * s2 / PBE_kappa;
        const double fx = 1.0 + PBE_kappa - PBE_kappa / denom;

        // Energy density: exc = ex_LDA * F_x
        const double ex_lda = ax * n13;
        exc[i] = ex_lda * fx;

        // d(F_x)/d(s^2) = mu / denom^2
        const double dfx_ds2 = PBE_mu / (denom * denom);

        // d(s^2)/dn = -(8/3) * s^2 / n  (since s^2 ~ |nabla n|^2 / (4*kf^2*n^2)
        //    and kf ~ n^{1/3}, so s^2 ~ sigma * n^{-8/3} * const)
        // vrho = d(n * exc)/dn = exc + n * d(exc)/dn
        //      = (4/3)*ex_lda*fx + ex_lda * n * dfx_ds2 * ds2_dn
        // ds2_dn = sigma * d/dn [1/(4*kf^2*n^2)] = sigma * (-8/3) / (4*kf^2*n^3)
        //        = -(8/3) * s^2 / n
        vrho[i] = (4.0 / 3.0) * ex_lda * (fx - s2 * dfx_ds2 * (8.0 / 3.0));

        // vsigma = d(n*exc)/d(sigma) = n * ex_lda * dfx_ds2 * ds2_dsigma
        // ds2_dsigma = 1 / (4*kf^2*n^2)
        if (kf > 1e-30) {
            vsigma[i] = ex_lda * dfx_ds2 / (4.0 * kf * kf * n);
        } else {
            vsigma[i] = 0.0;
        }
    }
}

/// Compute unpolarized PBE correlation energy density and potentials.
/// Uses PZ LDA correlation as the base, adds gradient correction H(rs,t).
/// All in Rydberg units.
static void pbe_c_unpol(const double* rho, const double* sigma,
                         double* ec_out, double* vrho_c, double* vsigma_c, int np)
{
    static const double rs_prefactor = std::cbrt(3.0 / (4.0 * constants::pi));

    for (int i = 0; i < np; ++i) {
        const double n = std::max(rho[i], 1e-30);
        const double sig = std::max(sigma[i], 0.0);

        const double rs = rs_prefactor / std::cbrt(n);
        const double kf = std::cbrt(3.0 * constants::pi * constants::pi * n);
        const double ks = std::sqrt(4.0 * kf / constants::pi);
        const double gn = std::sqrt(sig);

        // t = |nabla n| / (2 * ks * n)
        const double t = (ks > 1e-30 && n > 1e-30) ? gn / (2.0 * ks * n) : 0.0;
        const double t2 = t * t;

        // LDA correlation (PZ parameterization, Ry units)
        double ec_lda, vc_lda;
        {
            // PZ parameters (Rydberg): unpolarized
            constexpr double gamma_pz = -0.2846;
            constexpr double beta1 = 1.0529;
            constexpr double beta2 = 0.3334;
            constexpr double A = 0.0622;
            constexpr double B = -0.0960;
            constexpr double C = 0.0040;
            constexpr double D = -0.0232;

            if (rs >= 1.0) {
                double sqrs = std::sqrt(rs);
                double denom = 1.0 + beta1 * sqrs + beta2 * rs;
                ec_lda = gamma_pz / denom;
                double dec_drs = -gamma_pz * (0.5 * beta1 / sqrs + beta2) / (denom * denom);
                vc_lda = ec_lda - (rs / 3.0) * dec_drs;
            } else {
                double lnrs = std::log(rs);
                ec_lda = A * lnrs + B + C * rs * lnrs + D * rs;
                double dec_drs = A / rs + C * lnrs + C + D;
                vc_lda = ec_lda - (rs / 3.0) * dec_drs;
            }
        }

        // PBE gradient correction H(t)
        // A = beta/gamma / (exp(-ec_lda/(gamma)) - 1)
        // Note: ec_lda is in Ry, gamma is in Ry as well (already in Ry)
        double exp_arg = -ec_lda / PBE_gamma;
        double exp_val = std::exp(exp_arg);
        double A_pbe = PBE_beta / PBE_gamma / std::max(exp_val - 1.0, 1e-30);

        double At2 = A_pbe * t2;
        double numer = 1.0 + At2;
        double denom_h = 1.0 + At2 + A_pbe * A_pbe * t2 * t2;
        double frac = (denom_h > 1e-30) ? numer / denom_h : 0.0;

        double H = PBE_gamma * std::log(1.0 + PBE_beta / PBE_gamma * t2 * frac);

        ec_out[i] = ec_lda + H;

        // For simplicity in the built-in version, we use numerical stability
        // but approximate the vrho and vsigma contributions
        // dH/dt^2 chain rule for vrho and vsigma

        // dH/d(t^2):
        double inner = PBE_beta / PBE_gamma * t2 * frac;
        double d_inner;  // d(t^2*frac)/d(t^2)
        {
            // d(t^2 * numer/denom)/d(t^2)
            // = numer/denom + t^2 * d(numer/denom)/d(t^2)
            // numer = 1 + A*t^2, denom = 1 + A*t^2 + A^2*t^4
            // d(numer)/d(t^2) = A
            // d(denom)/d(t^2) = A + 2*A^2*t^2
            double dn_dt2 = A_pbe;
            double dd_dt2 = A_pbe + 2.0 * A_pbe * A_pbe * t2;
            d_inner = (dn_dt2 * denom_h - numer * dd_dt2) / (denom_h * denom_h);
            d_inner = frac + t2 * d_inner;
        }
        double dH_dt2 = PBE_gamma / (1.0 + inner) * PBE_beta / PBE_gamma * d_inner;

        // t^2 = sigma / (4 * ks^2 * n^2), ks = sqrt(4*kf/pi), kf = (3*pi^2*n)^{1/3}
        // dt^2/dsigma = 1/(4*ks^2*n^2)
        // dt^2/dn = sigma * d/dn[1/(4*ks^2*n^2)]
        //         = -(7/3) * t^2 / n  (since ks ~ n^{1/6}, so ks^2*n^2 ~ n^{7/3})

        if (ks > 1e-30 && n > 1e-30) {
            vsigma_c[i] = dH_dt2 / (4.0 * ks * ks * n * n);
        } else {
            vsigma_c[i] = 0.0;
        }

        // vrho_c = vc_lda + H + n * dH/dn
        // dH/dn = dH/dt^2 * dt^2/dn + dH/dA * dA/dn
        // For simplicity, main contribution:
        // dH/dn ≈ dH_dt2 * (-(7/3) * t^2 / n) + dH_dA_term
        // dH/dA is complex; we approximate by including the t^2 derivative only
        // (this is the dominant term)
        double dH_dn_approx = dH_dt2 * (-(7.0 / 3.0) * t2 / n);
        vrho_c[i] = vc_lda + H + n * dH_dn_approx;
    }
}

/// Spin-polarized PBE exchange: uses spin-scaling relation
/// E_x[n_up, n_dn] = (E_x[2*n_up] + E_x[2*n_dn]) / 2
static void pbe_x_spin(const double* rho_up, const double* rho_dn,
                        const double* sigma_uu, const double* sigma_dd,
                        double* exc, double* vrho_up, double* vrho_dn,
                        double* vsig_uu, double* vsig_dd, int np)
{
    // Exchange uses spin-scaling: exc(n_up, n_dn) = [n_up*ex(2*n_up,4*sig_uu) + n_dn*ex(2*n_dn,4*sig_dd)] / n
    // vxc_up = d[n*exc]/d(n_up) = vx(2*n_up) evaluated at 2*n_up
    // vsig_uu = d[n*exc]/d(sigma_uu) = vsig at 2*n_up with sigma=4*sigma_uu
    std::vector<double> rho_2u(np), rho_2d(np), sig_4uu(np), sig_4dd(np);
    for (int i = 0; i < np; ++i) {
        rho_2u[i] = 2.0 * std::max(rho_up[i], 0.0);
        rho_2d[i] = 2.0 * std::max(rho_dn[i], 0.0);
        sig_4uu[i] = 4.0 * std::max(sigma_uu[i], 0.0);
        sig_4dd[i] = 4.0 * std::max(sigma_dd[i], 0.0);
    }

    std::vector<double> ex_u(np), vr_u(np), vs_u(np);
    std::vector<double> ex_d(np), vr_d(np), vs_d(np);

    pbe_x_unpol(rho_2u.data(), sig_4uu.data(), ex_u.data(), vr_u.data(), vs_u.data(), np);
    pbe_x_unpol(rho_2d.data(), sig_4dd.data(), ex_d.data(), vr_d.data(), vs_d.data(), np);

    for (int i = 0; i < np; ++i) {
        double nu = std::max(rho_up[i], 0.0);
        double nd = std::max(rho_dn[i], 0.0);
        double n = nu + nd;
        if (n < 1e-30) {
            exc[i] = 0.0;
            vrho_up[i] = 0.0;
            vrho_dn[i] = 0.0;
            vsig_uu[i] = 0.0;
            vsig_dd[i] = 0.0;
            continue;
        }
        // Energy density: per total electron
        exc[i] = (nu * ex_u[i] + nd * ex_d[i]) / n;
        // Potentials: derivative of n*exc w.r.t. n_up = d[n_up * ex(2*n_up)]/d(n_up)
        // = ex(2*n_up) + 2*n_up * dex/d(2*n_up) = vx(2*n_up)
        vrho_up[i] = vr_u[i];
        vrho_dn[i] = vr_d[i];
        // vsigma_uu: d[n*exc]/d(sigma_uu) = n_up * dex_u/d(4*sigma_uu) * 4 / 2
        // The factor works out to 2*vsig evaluated at scaled args
        vsig_uu[i] = 2.0 * vs_u[i];
        vsig_dd[i] = 2.0 * vs_d[i];
    }
}

/// Spin-polarized PBE correlation
/// Uses VWN-like spin interpolation with PBE gradient correction
static void pbe_c_spin(const double* rho_up, const double* rho_dn,
                        const double* sigma_uu, const double* sigma_ud,
                        const double* sigma_dd,
                        double* ec_out, double* vrho_up, double* vrho_dn,
                        double* vsig_uu, double* vsig_ud, double* vsig_dd, int np)
{
    // For the built-in spin-polarized PBE correlation, we use a simplified
    // approach: compute the unpolarized PBE correlation with the total density
    // and total sigma = sigma_uu + 2*sigma_ud + sigma_dd, then apply the
    // LSDA spin-interpolation correction.
    // This is approximate but captures the main physics.

    static const double rs_prefactor = std::cbrt(3.0 / (4.0 * constants::pi));
    static const double f_denom = 2.0 * (std::cbrt(2.0) - 1.0);

    // PZ parameters (Rydberg): paramagnetic
    constexpr double gamma_p = -0.2846;
    constexpr double beta1_p = 1.0529;
    constexpr double beta2_p = 0.3334;
    constexpr double A_p = 0.0622;
    constexpr double B_p = -0.0960;
    constexpr double C_p = 0.0040;
    constexpr double D_p = -0.0232;

    // PZ parameters (Rydberg): ferromagnetic
    constexpr double gamma_f = -0.1686;
    constexpr double beta1_f = 1.3981;
    constexpr double beta2_f = 0.2611;
    constexpr double A_f = 0.0311;
    constexpr double B_f = -0.0538;
    constexpr double C_f = 0.0014;
    constexpr double D_f = -0.0096;

    auto ec_pz = [](double rs, double gamma, double beta1, double beta2,
                     double A, double B, double C, double D,
                     double& e_c, double& de_drs) {
        if (rs >= 1.0) {
            double sqrs = std::sqrt(rs);
            double denom = 1.0 + beta1 * sqrs + beta2 * rs;
            e_c = gamma / denom;
            de_drs = -gamma * (0.5 * beta1 / sqrs + beta2) / (denom * denom);
        } else {
            double lnrs = std::log(rs);
            e_c = A * lnrs + B + C * rs * lnrs + D * rs;
            de_drs = A / rs + C * lnrs + C + D;
        }
    };

    for (int i = 0; i < np; ++i) {
        double nu = std::max(rho_up[i], 0.0);
        double nd = std::max(rho_dn[i], 0.0);
        double n = nu + nd;

        if (n < 1e-30) {
            ec_out[i] = 0.0;
            vrho_up[i] = 0.0;
            vrho_dn[i] = 0.0;
            vsig_uu[i] = 0.0;
            vsig_ud[i] = 0.0;
            vsig_dd[i] = 0.0;
            continue;
        }

        double rs = rs_prefactor / std::cbrt(n);
        double zeta = (nu - nd) / n;
        zeta = std::max(-1.0 + 1e-12, std::min(1.0 - 1e-12, zeta));

        // f(zeta) spin interpolation
        double zp1 = 1.0 + zeta;
        double zm1 = 1.0 - zeta;
        double f_zeta = (std::cbrt(zp1 * zp1 * zp1 * zp1) +
                         std::cbrt(zm1 * zm1 * zm1 * zm1) - 2.0) / f_denom;
        double df_dzeta = (4.0 / 3.0) * (std::cbrt(zp1) - std::cbrt(zm1)) / f_denom;

        // LDA correlation: paramagnetic and ferromagnetic
        double ec_p, dec_p_drs, ec_f, dec_f_drs;
        ec_pz(rs, gamma_p, beta1_p, beta2_p, A_p, B_p, C_p, D_p, ec_p, dec_p_drs);
        ec_pz(rs, gamma_f, beta1_f, beta2_f, A_f, B_f, C_f, D_f, ec_f, dec_f_drs);

        double ec_lda = ec_p + (ec_f - ec_p) * f_zeta;

        // PBE gradient correction H(rs, zeta, t)
        double sig_total = std::max(sigma_uu[i], 0.0) + 2.0 * sigma_ud[i] + std::max(sigma_dd[i], 0.0);
        sig_total = std::max(sig_total, 0.0);

        double kf = std::cbrt(3.0 * constants::pi * constants::pi * n);
        double ks = std::sqrt(4.0 * kf / constants::pi);
        double gn = std::sqrt(sig_total);
        double t = (ks > 1e-30) ? gn / (2.0 * ks * n) : 0.0;
        double t2 = t * t;

        // Spin-scaling of ks: phi(zeta) = [(1+zeta)^{2/3} + (1-zeta)^{2/3}] / 2
        double phi = 0.5 * (std::cbrt(zp1 * zp1) + std::cbrt(zm1 * zm1));
        double phi3 = phi * phi * phi;

        // Scaled t: t_scaled = t / phi
        double t_s = (phi > 1e-30) ? t / phi : 0.0;
        double t_s2 = t_s * t_s;

        // A_pbe = beta/gamma / (exp(-ec_lda/(gamma*phi^3)) - 1)
        double exp_arg = -ec_lda / (PBE_gamma * phi3);
        double exp_val = std::exp(std::min(exp_arg, 500.0));
        double A_pbe = PBE_beta / PBE_gamma / std::max(exp_val - 1.0, 1e-30);

        double At2 = A_pbe * t_s2;
        double numer = 1.0 + At2;
        double denom_h = 1.0 + At2 + A_pbe * A_pbe * t_s2 * t_s2;
        double frac = (denom_h > 1e-30) ? numer / denom_h : 0.0;

        double H = PBE_gamma * phi3 * std::log(1.0 + PBE_beta / PBE_gamma * t_s2 * frac);

        ec_out[i] = ec_lda + H;

        // Approximate potentials via chain rule
        // For the built-in, we compute vrho from LDA spin + gradient correction
        double dec_drs = dec_p_drs + (dec_f_drs - dec_p_drs) * f_zeta;
        double vc_common = ec_lda - (rs / 3.0) * dec_drs;
        double delta_ec = ec_f - ec_p;

        vrho_up[i] = vc_common + delta_ec * df_dzeta * (1.0 - zeta);
        vrho_dn[i] = vc_common - delta_ec * df_dzeta * (1.0 + zeta);

        // Add gradient correction to vrho (approximate: H contribution)
        // dH/dt_s^2 term
        double inner_h = PBE_beta / PBE_gamma * t_s2 * frac;
        double dn_dt2 = A_pbe;
        double dd_dt2 = A_pbe + 2.0 * A_pbe * A_pbe * t_s2;
        double d_frac_dt2 = (dn_dt2 * denom_h - numer * dd_dt2) / (denom_h * denom_h);
        double d_inner = frac + t_s2 * d_frac_dt2;
        double dH_dts2 = PBE_gamma * phi3 / (1.0 + inner_h) * PBE_beta / PBE_gamma * d_inner;

        // dt_s^2/dn ≈ -(7/3) * t_s^2 / n (dominant term)
        double dH_dn_approx = dH_dts2 * (-(7.0 / 3.0) * t_s2 / n);
        vrho_up[i] += H + n * dH_dn_approx;
        vrho_dn[i] += H + n * dH_dn_approx;

        // vsigma: dt_s^2/dsigma = 1 / (4*ks^2*n^2*phi^2)
        double phi2 = phi * phi;
        if (ks > 1e-30 && n > 1e-30 && phi2 > 1e-30) {
            double dsig = dH_dts2 / (4.0 * ks * ks * n * n * phi2);
            // sigma_total = sigma_uu + 2*sigma_ud + sigma_dd
            // d(sigma_total)/d(sigma_uu) = 1
            // d(sigma_total)/d(sigma_ud) = 2
            // d(sigma_total)/d(sigma_dd) = 1
            vsig_uu[i] = dsig;
            vsig_ud[i] = 2.0 * dsig;
            vsig_dd[i] = dsig;
        } else {
            vsig_uu[i] = 0.0;
            vsig_ud[i] = 0.0;
            vsig_dd[i] = 0.0;
        }
    }
}

} // namespace builtin (pbe)

// ---------------------------------------------------------------------------
// Built-in unpolarized PBE evaluation
// ---------------------------------------------------------------------------

XCResult XCEvaluator::evaluate_builtin_pbe(const RVec& density_r,
                                             const RVec& sigma_r,
                                             double cell_volume) const
{
    const int np = static_cast<int>(density_r.size());

    XCResult result;
    result.exc.resize(np);
    result.vxc.resize(np);
    result.vsigma.resize(np);
    result.energy = 0.0;

    RVec ex(np), vr_x(np), vs_x(np);
    RVec ec(np), vr_c(np), vs_c(np);

    builtin::pbe_x_unpol(density_r.data(), sigma_r.data(),
                          ex.data(), vr_x.data(), vs_x.data(), np);
    builtin::pbe_c_unpol(density_r.data(), sigma_r.data(),
                          ec.data(), vr_c.data(), vs_c.data(), np);

    const double dv = cell_volume / static_cast<double>(np);
    double esum = 0.0;
    const double xs = exchange_scale_;
    for (int i = 0; i < np; ++i) {
        result.exc[i] = xs * ex[i] + ec[i];
        result.vxc[i] = xs * vr_x[i] + vr_c[i];
        result.vsigma[i] = xs * vs_x[i] + vs_c[i];
        esum += result.exc[i] * std::max(density_r[i], 0.0);
    }
    result.energy = dv * esum;
    return result;
}

// ---------------------------------------------------------------------------
// Built-in spin-polarized PBE evaluation
// ---------------------------------------------------------------------------

XCEvaluator::SpinGGAResult XCEvaluator::evaluate_builtin_spin_pbe(
    const RVec& density_up,
    const RVec& density_dn,
    const RVec& sigma_uu,
    const RVec& sigma_ud,
    const RVec& sigma_dd,
    double cell_volume) const
{
    const int np = static_cast<int>(density_up.size());

    SpinGGAResult result;
    result.vxc_up.resize(np);
    result.vxc_dn.resize(np);
    result.vsigma_uu.resize(np);
    result.vsigma_ud.resize(np);
    result.vsigma_dd.resize(np);
    result.energy = 0.0;

    // Exchange
    RVec ex(np), vr_x_up(np), vr_x_dn(np), vs_x_uu(np), vs_x_dd(np);
    builtin::pbe_x_spin(density_up.data(), density_dn.data(),
                         sigma_uu.data(), sigma_dd.data(),
                         ex.data(), vr_x_up.data(), vr_x_dn.data(),
                         vs_x_uu.data(), vs_x_dd.data(), np);

    // Correlation
    RVec ec(np), vr_c_up(np), vr_c_dn(np), vs_c_uu(np), vs_c_ud(np), vs_c_dd(np);
    builtin::pbe_c_spin(density_up.data(), density_dn.data(),
                         sigma_uu.data(), sigma_ud.data(), sigma_dd.data(),
                         ec.data(), vr_c_up.data(), vr_c_dn.data(),
                         vs_c_uu.data(), vs_c_ud.data(), vs_c_dd.data(), np);

    const double dv = cell_volume / static_cast<double>(np);
    double esum = 0.0;
    const double xs = exchange_scale_;
    for (int i = 0; i < np; ++i) {
        double exc = xs * ex[i] + ec[i];
        double n = std::max(density_up[i], 0.0) + std::max(density_dn[i], 0.0);
        esum += exc * n;
        result.vxc_up[i] = xs * vr_x_up[i] + vr_c_up[i];
        result.vxc_dn[i] = xs * vr_x_dn[i] + vr_c_dn[i];
        result.vsigma_uu[i] = xs * vs_x_uu[i] + vs_c_uu[i];
        result.vsigma_ud[i] = vs_c_ud[i];  // exchange has no cross-spin sigma
        result.vsigma_dd[i] = xs * vs_x_dd[i] + vs_c_dd[i];
    }
    result.energy = dv * esum;
    return result;
}

} // namespace kronos
