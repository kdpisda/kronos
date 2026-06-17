---
title: Fermi-Dirac, Gaussian, and Methfessel-Paxton Smearing
description: Electronic temperature smearing in plane-wave DFT — Fermi-Dirac, Gaussian, and Methfessel-Paxton occupations, Brillouin-zone integration for metals, and -TS entropy correction.
keywords:
  - Fermi-Dirac smearing
  - Gaussian smearing
  - Methfessel-Paxton smearing
  - Brillouin-zone integration
  - metallic DFT
  - Fermi level bisection
  - DFT entropy
  - electronic temperature
  - occupations DFT
  - plane-wave DFT metals
slug: /physics/smearing
sidebar_position: 10
---

# Fermi-Dirac, Gaussian, and Methfessel-Paxton Smearing

In a metal, the Brillouin-zone integral that defines the electron density runs across a Fermi surface where band occupations change abruptly between 1 and 0. A naive sampling of this integral on a finite k-mesh converges painfully slowly — and the convergence is non-monotonic, oscillating with the mesh density. Smearing replaces the sharp step in occupation by a smooth function with a controllable width, transforming a hard integration problem into a tame one. The price is a small fictitious electronic temperature that needs to be subtracted (the "$-TS$" entropy term) to recover the true ground-state total energy.

## Why smearing is necessary

The electron density of a periodic system is a Brillouin-zone integral:

$$n(\mathbf{r}) = \sum_{n} \int_\mathrm{BZ} \frac{d\mathbf{k}}{(2\pi)^3}\, f_{n\mathbf{k}}\, |\psi_{n\mathbf{k}}(\mathbf{r})|^2$$

where $f_{n\mathbf{k}} = \Theta(\varepsilon_F - \varepsilon_{n\mathbf{k}})$ at zero temperature. For an insulator this is harmless: $f$ is 1 or 0 everywhere except across the gap, and any reasonable k-mesh captures the integral well. In a metal, the Fermi surface crosses bands; near it, $f$ jumps from 1 to 0 over an infinitesimal range of $\mathbf{k}$. Sampling this with a discrete k-grid gives an integrand that is discontinuous in $\mathbf{k}$, and the discretization error decays as $O(1/N_k)$ — far worse than smooth integrands where Gauss-Legendre-style schemes give exponential convergence.

Smearing broadens the step function into a smooth $f(\varepsilon)$ with a width $\sigma$. The integrand becomes smooth, k-mesh integration converges fast, and we recover the zero-temperature result in the $\sigma \to 0$ limit (in practice: extrapolation, or a small enough $\sigma$ that the residual error is below the target accuracy).

## Fermi-Dirac smearing

The physically-motivated choice is the Fermi-Dirac distribution at electronic temperature $T$:

$$f_\mathrm{FD}(\varepsilon) = \frac{1}{1 + \exp\!\left( \dfrac{\varepsilon - \varepsilon_F}{k_B T} \right)}$$

with smearing width $\sigma = k_B T$. As $T \to 0$, $f_\mathrm{FD} \to \Theta(\varepsilon_F - \varepsilon)$, recovering the step function. Fermi-Dirac is the right physical model for a real metal at finite temperature, and it gives the textbook entropy term

$$S_\mathrm{el}^\mathrm{FD} = -k_B \sum_{n\mathbf{k}} w_\mathbf{k}\, \big[ f \ln f + (1-f) \ln(1-f) \big]$$

The main limitation is that the tails of $f_\mathrm{FD}$ decay only exponentially — at $|\varepsilon - \varepsilon_F| = 10\sigma$ the residual occupation is still $\sim 5 \times 10^{-5}$. For high-accuracy total-energy calculations you must integrate further from $\varepsilon_F$ than for the alternatives below.

## Gaussian smearing

A faster-decaying alternative is the complementary error function:

$$f_\mathrm{G}(\varepsilon) = \tfrac{1}{2}\,\mathrm{erfc}\!\left( \frac{\varepsilon - \varepsilon_F}{\sigma} \right)$$

The derivative is a Gaussian, $-(1/\sigma\sqrt{\pi})\, e^{-x^2}$ with $x = (\varepsilon-\varepsilon_F)/\sigma$. The Gaussian decays so fast that at $|x| = 4$ it is already at $10^{-7}$, so the BZ integral is exquisitely localized around $\varepsilon_F$. Gaussian smearing has no physical interpretation — it does not correspond to any real ensemble — but it is the workhorse for variational SCF total-energy calculations because of its excellent convergence properties.

The entropy correction for Gaussian smearing (in the convention used by Quantum ESPRESSO and by KRONOS for compatibility) is

$$S_\mathrm{el}^\mathrm{G} = -\frac{\sigma}{2\sqrt{\pi}}\, \sum_{n\mathbf{k}} w_\mathbf{k}\, e^{-x_{n\mathbf{k}}^2}, \quad x_{n\mathbf{k}} = \frac{\varepsilon_{n\mathbf{k}} - \varepsilon_F}{\sigma}$$

with $TS$ folded into the prefactor. KRONOS matches the QE convention exactly so that numerical comparisons are like-for-like.

## Methfessel-Paxton smearing

Methfessel & Paxton (1989) constructed a family of higher-order smearing functions that integrate polynomials exactly up to degree $2N+1$. Their order-$N$ function adds Hermite-polynomial corrections to the Gaussian:

$$f_N(\varepsilon) = f_\mathrm{G}(\varepsilon) + \sum_{n=1}^{N} A_n\, H_{2n-1}\!\left(\frac{\varepsilon - \varepsilon_F}{\sigma}\right) e^{-((\varepsilon - \varepsilon_F)/\sigma)^2}$$

with $A_n = (-1)^n / (n!\, 4^n\, \sqrt{\pi})$ and $H_{2n-1}$ the physicists' Hermite polynomials. The MP1 (N=1) correction subtracts a small overshoot in the Gaussian's $f$ at $\varepsilon = \varepsilon_F$. The benefit is that the entropy contribution at converged $\sigma$ is much smaller — for the same physical accuracy you can use a 2–3× larger $\sigma$ than Gaussian, and integrate the BZ on a smaller k-mesh.

The trade-off is that $f_\mathrm{MP1}(\varepsilon)$ is **not monotonic** in $\varepsilon$ and can briefly exceed 1 or drop below 0. This is harmless for total-energy work but breaks the interpretation of "occupation," so MP smearing should not be used for density-of-states plots without smoothing. KRONOS uses MP1 by default for metals when `smearing: mp` is requested.

## Fermi level by bisection

Given target electron count $N_\mathrm{electrons}$, the Fermi level is the solution of

$$\sum_{n\mathbf{k}} w_\mathbf{k}\, f(\varepsilon_{n\mathbf{k}};\, \varepsilon_F) = N_\mathrm{electrons}$$

where $w_\mathbf{k}$ are the k-point weights (including factor of 2 for spin if `nspin: 1`). KRONOS solves this by bisection on $\varepsilon_F$. The natural bracketing is $[\varepsilon_\min, \varepsilon_\max]$ over all band eigenvalues; bisection converges in $\log_2((\varepsilon_\max - \varepsilon_\min)/\epsilon)$ iterations.

The bisection tolerance is set to $\epsilon = 10^{-10}\,\mathrm{Ry}$ — **not** tighter. A tighter tolerance causes SCF oscillation with `smearing: None` on toy pseudopotentials, because the Fermi level shifts by floating-point noise between iterations and the occupations seesaw. (This is a real lesson learned during KRONOS development.)

## Entropy correction in the total energy

Smearing makes the variational quantity the Mermin free energy rather than the ground-state total energy:

$$F[n] = E[n] - T S_\mathrm{el}[n]$$

The Mermin functional is what KRONOS minimizes self-consistently. To report the *ground-state* total energy at the same density, KRONOS subtracts the entropy term:

$$E_\mathrm{tot}^{(T=0)} \approx F[n] + TS_\mathrm{el}[n]$$

This is the standard "Methfessel correction" used by all PW DFT codes. The output JSON reports both $F$ and $E_\mathrm{tot}^{(T=0)}$; the latter is the number to publish.

## Convergence with smearing width

The "right" $\sigma$ depends on the system. For typical metals:

- **Gaussian**: $\sigma = 0.01$–$0.03\,\mathrm{Ry}$ gives sub-meV/atom convergence with a moderate k-mesh.
- **Methfessel-Paxton 1**: $\sigma = 0.02$–$0.05\,\mathrm{Ry}$ for similar accuracy at coarser k-meshes.
- **Fermi-Dirac**: same as Gaussian but with slightly slower BZ-integration convergence; useful when finite-temperature physics matters (e.g., molten metals).

For insulators and semiconductors, set `smearing: None`. The Fermi level lands in the gap, no smearing is required, and the total energy is well-defined at $T = 0$.

## How KRONOS implements this

`FermiSolver::find_fermi_level()` in `src/solver/fermi.cpp` does the bisection. The smearing type and width come from the parsed `OccupationsParams` (YAML keys `smearing` and `smearing_width`). Occupations $f_{n\mathbf{k}}$ are stored alongside eigenvalues for density assembly and for the entropy correction. The entropy contribution is added to the total energy in `scf.cpp::compute_total_energy()`.

For multi-rank MPI runs (k-point parallelism), the eigenvalue arrays from all ranks are gathered before bisection so that the Fermi level is a global quantity.

## References

- Mermin, N. D. "Thermal properties of the inhomogeneous electron gas", *Phys. Rev.* **137**, A1441 (1965). — finite-$T$ DFT, Fermi-Dirac entropy.
- Methfessel, M. & Paxton, A. T. "High-precision sampling for Brillouin-zone integration in metals", *Phys. Rev. B* **40**, 3616 (1989).
- de Gironcoli, S. "Lattice dynamics of metals from density-functional perturbation theory", *Phys. Rev. B* **51**, 6773 (1995). — practical advice on choosing $\sigma$.
- Martin, R. M. *Electronic Structure*, Ch. 13.
- Quantum ESPRESSO documentation, `pw.x` input description — for the `degauss` convention KRONOS matches.
