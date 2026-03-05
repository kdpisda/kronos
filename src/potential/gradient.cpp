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

} // namespace kronos
