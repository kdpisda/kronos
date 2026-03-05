#pragma once

namespace kronos {

/// Compute the real spherical harmonic Y_lm at the direction given by (x, y, z).
///
/// The function normalizes (x, y, z) internally, so any non-zero vector can be
/// passed.  For the zero vector (|r| < 1e-30), only l=0 returns a nonzero
/// value: Y_00 = 1/sqrt(4*pi).  All other (l, m) return 0.
///
/// Supported range: l = 0..3, m = -l..+l.
/// Returns 0 for unsupported (l, m) combinations.
///
/// Convention (real solid harmonics, standard DFT / QE convention):
///
///   l=0:  Y_00     = 1/sqrt(4*pi)
///
///   l=1:  Y_1,-1   = sqrt(3/(4*pi)) * y/r
///         Y_1, 0   = sqrt(3/(4*pi)) * z/r
///         Y_1,+1   = sqrt(3/(4*pi)) * x/r
///
///   l=2:  Y_2,-2   = sqrt(15/(4*pi))   * xy/r^2
///         Y_2,-1   = sqrt(15/(4*pi))   * yz/r^2
///         Y_2, 0   = sqrt(5/(16*pi))   * (3z^2 - r^2)/r^2
///         Y_2,+1   = sqrt(15/(4*pi))   * xz/r^2
///         Y_2,+2   = sqrt(15/(16*pi))  * (x^2 - y^2)/r^2
///
///   l=3:  Y_3,-3   = sqrt(35/(32*pi))  * y(3x^2 - y^2) / r^3
///         Y_3,-2   = sqrt(105/(4*pi))  * xyz / r^3
///         Y_3,-1   = sqrt(21/(32*pi))  * y(5z^2 - r^2) / r^3
///         Y_3, 0   = sqrt(7/(16*pi))   * z(5z^2 - 3r^2) / r^3
///         Y_3,+1   = sqrt(21/(32*pi))  * x(5z^2 - r^2) / r^3
///         Y_3,+2   = sqrt(105/(16*pi)) * (x^2 - y^2)*z / r^3
///         Y_3,+3   = sqrt(35/(32*pi))  * x(x^2 - 3y^2) / r^3
///
double real_spherical_harmonic(int l, int m, double x, double y, double z);

} // namespace kronos
