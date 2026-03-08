#include "potential/gradient.hpp"

#include <cassert>
#include <cmath>
#include <vector>

namespace kronos {

// ---------------------------------------------------------------------------
// compute_sigma  --  |nabla n(r)|^2 from density in G-space
// ---------------------------------------------------------------------------

RVec compute_sigma(const CVec& density_g,
                   const PlaneWaveBasis& basis,
                   FFTGrid& fft_grid)
{
    const int num_pw = static_cast<int>(basis.num_pw());
    const int num_grid = fft_grid.total_points();
    const auto& gvecs = basis.gvectors();

    // We compute the gradient of n(r) in reciprocal space:
    //   (d n / d x_d)_G = i * G_d * n(G)
    // then IFFT each Cartesian component to real space, and accumulate
    //   sigma(r) = sum_d | d n / d x_d (r) |^2

    RVec sigma(num_grid, 0.0);

    // Loop over three Cartesian directions (d = 0,1,2 for x,y,z)
    for (int d = 0; d < 3; ++d) {
        // Build the G-space representation of i * G_d * n(G)
        CVec grad_g(num_pw);
        for (int ig = 0; ig < num_pw; ++ig) {
            double g_d = gvecs[ig].cart[d];
            // i * G_d * n(G)
            grad_g[ig] = complex_t(0.0, 1.0) * g_d * density_g[ig];
        }

        // Scatter onto full FFT grid
        std::vector<complex_t> grad_grid(num_grid, complex_t{0.0, 0.0});
        fft_grid.scatter_to_grid(basis, grad_g, grad_grid);

        // Inverse FFT to real space
        std::vector<complex_t> grad_r(num_grid);
        fft_grid.inverse(grad_grid, grad_r);

        // Accumulate |grad component|^2 into sigma
        for (int i = 0; i < num_grid; ++i) {
            // The real-space gradient component should be real for a real density,
            // but we use the full complex norm to be safe.
            sigma[i] += std::norm(grad_r[i]);  // |z|^2 = re^2 + im^2
        }
    }

    return sigma;
}

// ---------------------------------------------------------------------------
// compute_gga_potential  --  V_gga(r) = -2 * div( vsigma(r) * nabla n(r) )
// ---------------------------------------------------------------------------

RVec compute_gga_potential(const CVec& density_g,
                           const RVec& vsigma,
                           const PlaneWaveBasis& basis,
                           FFTGrid& fft_grid)
{
    const int num_pw = static_cast<int>(basis.num_pw());
    const int num_grid = fft_grid.total_points();
    const auto& gvecs = basis.gvectors();

    assert(static_cast<int>(vsigma.size()) == num_grid);

    // Step 1: Compute nabla n(r) components in real space.
    //         Also form h_d(r) = vsigma(r) * (d n / d x_d)(r) and FFT to G-space.
    //         Then accumulate div in G-space:
    //           div(G) = sum_d i * G_d * h_d(G)

    // Accumulator for the divergence in G-space (full FFT grid)
    std::vector<complex_t> div_g(num_grid, complex_t{0.0, 0.0});

    for (int d = 0; d < 3; ++d) {
        // Build i * G_d * n(G) in PW representation
        CVec grad_g_pw(num_pw);
        for (int ig = 0; ig < num_pw; ++ig) {
            double g_d = gvecs[ig].cart[d];
            grad_g_pw[ig] = complex_t(0.0, 1.0) * g_d * density_g[ig];
        }

        // Scatter to FFT grid and IFFT to real space
        std::vector<complex_t> grad_grid(num_grid, complex_t{0.0, 0.0});
        fft_grid.scatter_to_grid(basis, grad_g_pw, grad_grid);

        std::vector<complex_t> grad_r(num_grid);
        fft_grid.inverse(grad_grid, grad_r);

        // Form h_d(r) = vsigma(r) * (d n / d x_d)(r)
        std::vector<complex_t> h_r(num_grid);
        for (int i = 0; i < num_grid; ++i) {
            h_r[i] = vsigma[i] * grad_r[i];
        }

        // FFT h_d(r) to G-space
        std::vector<complex_t> h_g(num_grid);
        fft_grid.forward(h_r, h_g);

        // Accumulate i * G_d * h_d(G) into the divergence
        // We work on the full FFT grid. To get G_d for each grid point,
        // we scatter a delta-like array: but it is simpler to gather h_g
        // to PW representation, multiply by i*G_d, then scatter back.
        CVec h_g_pw(num_pw);
        fft_grid.gather_from_grid(basis, h_g, h_g_pw);

        CVec div_component(num_pw);
        for (int ig = 0; ig < num_pw; ++ig) {
            double g_d = gvecs[ig].cart[d];
            div_component[ig] = complex_t(0.0, 1.0) * g_d * h_g_pw[ig];
        }

        // Scatter back to full grid and accumulate
        std::vector<complex_t> div_component_grid(num_grid, complex_t{0.0, 0.0});
        fft_grid.scatter_to_grid(basis, div_component, div_component_grid);
        for (int i = 0; i < num_grid; ++i) {
            div_g[i] += div_component_grid[i];
        }
    }

    // Step 2: IFFT the divergence to real space
    std::vector<complex_t> div_r(num_grid);
    fft_grid.inverse(div_g, div_r);

    // Step 3: V_gga(r) = -2 * div(h)(r)
    RVec vgga(num_grid);
    for (int i = 0; i < num_grid; ++i) {
        vgga[i] = -2.0 * div_r[i].real();
    }

    return vgga;
}

// ---------------------------------------------------------------------------
// compute_spin_sigma  --  per-spin |nabla n_s(r)|^2 and cross-term
// ---------------------------------------------------------------------------

SpinSigmaResult compute_spin_sigma(const CVec& density_up_g,
                                    const CVec& density_dn_g,
                                    const PlaneWaveBasis& basis,
                                    FFTGrid& fft_grid)
{
    const int num_pw = static_cast<int>(basis.num_pw());
    const int num_grid = fft_grid.total_points();
    const auto& gvecs = basis.gvectors();

    SpinSigmaResult result;
    result.sigma_uu.assign(num_grid, 0.0);
    result.sigma_ud.assign(num_grid, 0.0);
    result.sigma_dd.assign(num_grid, 0.0);
    result.grad_r.resize(2, std::vector<RVec>(3, RVec(num_grid, 0.0)));

    // For each spin, compute the 3 gradient components in real space.
    const CVec* density_g_spin[2] = {&density_up_g, &density_dn_g};

    for (int ispin = 0; ispin < 2; ++ispin) {
        const CVec& dens_g = *density_g_spin[ispin];
        for (int d = 0; d < 3; ++d) {
            // Build i * G_d * n_s(G) in PW representation
            CVec grad_g(num_pw);
            for (int ig = 0; ig < num_pw; ++ig) {
                double g_d = gvecs[ig].cart[d];
                grad_g[ig] = complex_t(0.0, 1.0) * g_d * dens_g[ig];
            }

            // Scatter onto full FFT grid
            std::vector<complex_t> grad_grid(num_grid, complex_t{0.0, 0.0});
            fft_grid.scatter_to_grid(basis, grad_g, grad_grid);

            // Inverse FFT to real space
            std::vector<complex_t> grad_r_c(num_grid);
            fft_grid.inverse(grad_grid, grad_r_c);

            // Store real part
            for (int i = 0; i < num_grid; ++i) {
                result.grad_r[ispin][d][i] = grad_r_c[i].real();
            }
        }
    }

    // Compute sigma components from gradients
    for (int i = 0; i < num_grid; ++i) {
        double suu = 0.0, sud = 0.0, sdd = 0.0;
        for (int d = 0; d < 3; ++d) {
            double gu = result.grad_r[0][d][i];
            double gd = result.grad_r[1][d][i];
            suu += gu * gu;
            sud += gu * gd;
            sdd += gd * gd;
        }
        result.sigma_uu[i] = suu;
        result.sigma_ud[i] = sud;
        result.sigma_dd[i] = sdd;
    }

    return result;
}

// ---------------------------------------------------------------------------
// compute_spin_gga_potential  --  per-spin GGA potential corrections
//
//   V_gga_up(r) = -2 * div(vsigma_uu * nabla n_up + vsigma_ud * nabla n_dn)
//   V_gga_dn(r) = -2 * div(vsigma_dd * nabla n_dn + vsigma_ud * nabla n_up)
// ---------------------------------------------------------------------------

void compute_spin_gga_potential(const CVec& density_up_g,
                                const CVec& density_dn_g,
                                const RVec& vsigma_uu,
                                const RVec& vsigma_ud,
                                const RVec& vsigma_dd,
                                const SpinSigmaResult& spin_sigma,
                                const PlaneWaveBasis& basis,
                                FFTGrid& fft_grid,
                                RVec& vgga_up,
                                RVec& vgga_dn)
{
    const int num_pw = static_cast<int>(basis.num_pw());
    const int num_grid = fft_grid.total_points();
    const auto& gvecs = basis.gvectors();

    // We compute both spin potentials in one pass over the directions.
    // For spin-up:
    //   h_up_d(r) = vsigma_uu(r) * grad_up_d(r) + vsigma_ud(r) * grad_dn_d(r)
    // For spin-down:
    //   h_dn_d(r) = vsigma_dd(r) * grad_dn_d(r) + vsigma_ud(r) * grad_up_d(r)
    //
    // div_s(G) = sum_d i * G_d * h_s_d(G)
    // V_gga_s(r) = -2 * IFFT(div_s(G))

    // Accumulators for divergence in G-space (full FFT grid)
    std::vector<complex_t> div_up_g(num_grid, complex_t{0.0, 0.0});
    std::vector<complex_t> div_dn_g(num_grid, complex_t{0.0, 0.0});

    for (int d = 0; d < 3; ++d) {
        // Build h_up_d(r) and h_dn_d(r) from precomputed gradients
        std::vector<complex_t> h_up_r(num_grid);
        std::vector<complex_t> h_dn_r(num_grid);
        for (int i = 0; i < num_grid; ++i) {
            double gu_d = spin_sigma.grad_r[0][d][i];
            double gd_d = spin_sigma.grad_r[1][d][i];
            h_up_r[i] = complex_t(vsigma_uu[i] * gu_d + vsigma_ud[i] * gd_d, 0.0);
            h_dn_r[i] = complex_t(vsigma_dd[i] * gd_d + vsigma_ud[i] * gu_d, 0.0);
        }

        // FFT h_s_d(r) -> h_s_d(G) and accumulate divergence
        for (int ispin = 0; ispin < 2; ++ispin) {
            auto& h_r = (ispin == 0) ? h_up_r : h_dn_r;
            auto& div_g = (ispin == 0) ? div_up_g : div_dn_g;

            std::vector<complex_t> h_g(num_grid);
            fft_grid.forward(h_r, h_g);

            // Gather to PW, multiply by i*G_d, scatter back
            CVec h_g_pw(num_pw);
            fft_grid.gather_from_grid(basis, h_g, h_g_pw);

            CVec div_component(num_pw);
            for (int ig = 0; ig < num_pw; ++ig) {
                double g_d = gvecs[ig].cart[d];
                div_component[ig] = complex_t(0.0, 1.0) * g_d * h_g_pw[ig];
            }

            std::vector<complex_t> div_component_grid(num_grid, complex_t{0.0, 0.0});
            fft_grid.scatter_to_grid(basis, div_component, div_component_grid);
            for (int i = 0; i < num_grid; ++i) {
                div_g[i] += div_component_grid[i];
            }
        }
    }

    // IFFT divergences and compute V_gga = -2 * div(h)
    vgga_up.resize(num_grid);
    vgga_dn.resize(num_grid);

    std::vector<complex_t> div_up_r(num_grid), div_dn_r(num_grid);
    fft_grid.inverse(div_up_g, div_up_r);
    fft_grid.inverse(div_dn_g, div_dn_r);

    for (int i = 0; i < num_grid; ++i) {
        vgga_up[i] = -2.0 * div_up_r[i].real();
        vgga_dn[i] = -2.0 * div_dn_r[i].real();
    }
}

} // namespace kronos
