#include "hamiltonian/hamiltonian.hpp"
#include "utils/timer.hpp"
#include "utils/logger.hpp"
#include <cassert>
#include <cmath>

namespace kronos {

Hamiltonian::Hamiltonian(const Crystal& crystal,
                         const PlaneWaveBasis& basis,
                         FFTGrid& fft_grid,
                         const NonlocalPP& nonlocal_pp)
    : crystal_(crystal)
    , basis_(basis)
    , fft_grid_(fft_grid)
    , nonlocal_pp_(nonlocal_pp)
    , veff_r_(fft_grid.total_points(), complex_t{0.0, 0.0})
{
}

void Hamiltonian::update_veff(const std::vector<complex_t>& veff_r) {
    assert(static_cast<int>(veff_r.size()) == fft_grid_.total_points());
    veff_r_ = veff_r;
}

CVec Hamiltonian::apply(const CVec& psi_g, const Vec3& k_frac) const {
    KRONOS_TIMER("hamiltonian_apply");

    const size_t npw = psi_g.size();
    const int total_points = fft_grid_.total_points();

    // 1. Kinetic energy: T|psi>_G = |k+G|^2 / 2 * psi_G  (factor of 2 for Ry units)
    //    kinetic_energies() already returns |k+G|^2/2 in Ry
    auto ke = basis_.kinetic_energies(k_frac);
    CVec hpsi(npw);
    for (size_t i = 0; i < npw; ++i) {
        hpsi[i] = ke[i] * psi_g[i];
    }

    // 2. Local potential: V_eff|psi> via FFT
    {
        KRONOS_TIMER("hamiltonian_fft");

        // a. Scatter psi_G onto the full FFT grid
        std::vector<complex_t> grid(total_points, complex_t{0.0, 0.0});
        fft_grid_.scatter_to_grid(basis_, psi_g, grid);

        // b. Inverse FFT: G-space -> real-space psi(r)
        std::vector<complex_t> psi_r(total_points);
        fft_grid_.inverse(grid, psi_r);

        // c. Multiply in real space: (V_eff * psi)(r)
        for (int i = 0; i < total_points; ++i) {
            psi_r[i] *= veff_r_[i];
        }

        // d. Forward FFT: real-space -> G-space
        fft_grid_.forward(psi_r, grid);

        // e. Gather back to PW coefficients
        CVec vloc_psi(basis_.num_pw());
        fft_grid_.gather_from_grid(basis_, grid, vloc_psi);

        // f. Add to hpsi
        for (size_t i = 0; i < npw; ++i) {
            hpsi[i] += vloc_psi[i];
        }
    }

    // 3. Non-local PP: V_NL|psi>
    {
        KRONOS_TIMER("hamiltonian_nonlocal");
        CVec vnl_psi = nonlocal_pp_.apply(psi_g, k_frac);
        for (size_t i = 0; i < npw; ++i) {
            hpsi[i] += vnl_psi[i];
        }
    }

    return hpsi;
}

std::function<CVec(const CVec&)> Hamiltonian::get_apply_function(const Vec3& k_frac) const {
    return [this, k_frac](const CVec& psi_g) -> CVec {
        return this->apply(psi_g, k_frac);
    };
}

std::vector<double> Hamiltonian::kinetic_diagonal(const Vec3& k_frac) const {
    return basis_.kinetic_energies(k_frac);
}

} // namespace kronos
