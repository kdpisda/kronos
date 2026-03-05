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
    // No libxc -- fall back to built-in LDA and set vsigma to zero
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

    // Sum exchange + correlation
    const double dv = cell_volume / static_cast<double>(np);
    double esum = 0.0;

    for (int i = 0; i < np; ++i) {
        result.exc[i] = ex[i] + ec[i];
        result.vxc[i] = vx[i] + vc[i];
        esum += result.exc[i] * std::max(density_r[i], 0.0);
    }

    result.energy = dv * esum;
    return result;
}

} // namespace kronos
