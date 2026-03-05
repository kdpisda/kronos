#include "core/spherical_harmonics.hpp"
#include "core/constants.hpp"

#include <cmath>

namespace kronos {

double real_spherical_harmonic(int l, int m, double x, double y, double z)
{
    const double r2 = x * x + y * y + z * z;
    const double r  = std::sqrt(r2);

    // Zero vector: only l=0 contributes
    if (r < 1.0e-30) {
        if (l == 0 && m == 0) {
            return 1.0 / std::sqrt(constants::four_pi);
        }
        return 0.0;
    }

    // Normalize to unit vector
    const double inv_r = 1.0 / r;
    const double ux = x * inv_r;
    const double uy = y * inv_r;
    const double uz = z * inv_r;

    switch (l) {
    case 0:
        // Y_00 = 1/sqrt(4*pi)
        if (m == 0) {
            return 1.0 / std::sqrt(constants::four_pi);
        }
        break;

    case 1:
        switch (m) {
        case -1: // Y_1,-1 = sqrt(3/(4*pi)) * y/r
            return std::sqrt(3.0 / constants::four_pi) * uy;
        case  0: // Y_1,0  = sqrt(3/(4*pi)) * z/r
            return std::sqrt(3.0 / constants::four_pi) * uz;
        case  1: // Y_1,+1 = sqrt(3/(4*pi)) * x/r
            return std::sqrt(3.0 / constants::four_pi) * ux;
        default:
            break;
        }
        break;

    case 2:
        switch (m) {
        case -2: // Y_2,-2 = sqrt(15/(4*pi)) * xy/r^2
            return std::sqrt(15.0 / (4.0 * constants::pi)) * ux * uy;
        case -1: // Y_2,-1 = sqrt(15/(4*pi)) * yz/r^2
            return std::sqrt(15.0 / (4.0 * constants::pi)) * uy * uz;
        case  0: // Y_2,0  = sqrt(5/(16*pi)) * (3z^2 - r^2)/r^2
            return std::sqrt(5.0 / (16.0 * constants::pi)) * (3.0 * uz * uz - 1.0);
        case  1: // Y_2,+1 = sqrt(15/(4*pi)) * xz/r^2
            return std::sqrt(15.0 / (4.0 * constants::pi)) * ux * uz;
        case  2: // Y_2,+2 = sqrt(15/(16*pi)) * (x^2 - y^2)/r^2
            return std::sqrt(15.0 / (16.0 * constants::pi)) * (ux * ux - uy * uy);
        default:
            break;
        }
        break;

    case 3:
        switch (m) {
        case -3: // Y_3,-3 = sqrt(35/(32*pi)) * y(3x^2 - y^2)/r^3
            return std::sqrt(35.0 / (32.0 * constants::pi))
                   * uy * (3.0 * ux * ux - uy * uy);
        case -2: // Y_3,-2 = sqrt(105/(4*pi)) * xyz/r^3
            return std::sqrt(105.0 / (4.0 * constants::pi))
                   * ux * uy * uz;
        case -1: // Y_3,-1 = sqrt(21/(32*pi)) * y(5z^2 - r^2)/r^3
            return std::sqrt(21.0 / (32.0 * constants::pi))
                   * uy * (5.0 * uz * uz - 1.0);
        case  0: // Y_3,0  = sqrt(7/(16*pi)) * z(5z^2 - 3r^2)/r^3
            return std::sqrt(7.0 / (16.0 * constants::pi))
                   * uz * (5.0 * uz * uz - 3.0);
        case  1: // Y_3,+1 = sqrt(21/(32*pi)) * x(5z^2 - r^2)/r^3
            return std::sqrt(21.0 / (32.0 * constants::pi))
                   * ux * (5.0 * uz * uz - 1.0);
        case  2: // Y_3,+2 = sqrt(105/(16*pi)) * (x^2 - y^2)*z/r^3
            return std::sqrt(105.0 / (16.0 * constants::pi))
                   * (ux * ux - uy * uy) * uz;
        case  3: // Y_3,+3 = sqrt(35/(32*pi)) * x(x^2 - 3y^2)/r^3
            return std::sqrt(35.0 / (32.0 * constants::pi))
                   * ux * (ux * ux - 3.0 * uy * uy);
        default:
            break;
        }
        break;

    default:
        break;
    }

    return 0.0;
}

} // namespace kronos
