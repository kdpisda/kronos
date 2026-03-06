#pragma once
#include <vector>

namespace kronos {

/// Simpson's rule integration for radial grids (matches QE convention).
/// Integrates func[i] * rab[i] using the 1-4-2-4-...-4-1 pattern.
inline double simpson_radial(const double* func, const double* rab, int npts) {
    if (npts < 3) {
        double s = 0.0;
        for (int i = 0; i < npts; ++i) s += func[i] * rab[i];
        return s;
    }
    // Ensure odd number of points for Simpson's rule
    int mesh = (npts % 2 == 0) ? npts - 1 : npts;

    double asum = 0.0;
    double f3 = func[0] * rab[0] / 3.0;
    for (int i = 1; i < mesh - 1; i += 2) {
        double f1 = f3;
        double f2 = func[i] * rab[i] / 3.0;
        f3 = func[i + 1] * rab[i + 1] / 3.0;
        asum += f1 + 4.0 * f2 + f3;
    }
    // If npts was even, add last point with trapezoidal
    if (npts % 2 == 0) {
        asum += 0.5 * (func[npts - 2] * rab[npts - 2]
                      + func[npts - 1] * rab[npts - 1]);
    }
    return asum;
}

/// Convenience overload for std::vector
inline double simpson_radial(const std::vector<double>& func,
                             const std::vector<double>& rab, int npts) {
    return simpson_radial(func.data(), rab.data(), npts);
}

} // namespace kronos
