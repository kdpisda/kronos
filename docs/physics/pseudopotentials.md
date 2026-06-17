---
title: Pseudopotentials in Plane-Wave DFT
description: Norm-conserving and PAW pseudopotentials, Kleinman-Bylander separable form, Coulomb tail subtraction, UPF format — how KRONOS loads and applies them.
keywords:
  - pseudopotential DFT
  - norm-conserving pseudopotential
  - Kleinman-Bylander
  - PAW pseudopotential
  - UPF format
  - DFT projectors
  - ONCV pseudopotential
  - plane-wave DFT pseudopotentials
  - V_loc
  - nonlocal projector
slug: /physics/pseudopotentials
sidebar_position: 7
---

In plane-wave density functional theory, treating all electrons—core and valence—explicitly requires enormous kinetic energy cutoffs to resolve the rapid oscillations of core wavefunctions near the nucleus. Pseudopotentials solve this by freezing core electrons and replacing their effect with a smooth, effective potential that acts only on valence electrons. Outside a cutoff radius $r_c$, the pseudo-wavefunction matches the all-electron (AE) wavefunction exactly; inside, it is smooth and nodeless. This dramatically reduces the plane-wave basis size—from thousands of components to hundreds—without sacrificing accuracy.

KRONOS supports both **norm-conserving pseudopotentials** (NC-PP) and **projector-augmented wave** (PAW) potentials via the Unified Pseudopotential Format (UPF v2). This page explains the physics, the Kleinman-Bylander separable representation, and how KRONOS evaluates pseudopotential contributions to the Hamiltonian.

## Why Pseudopotentials: The Core Problem

Core electrons are tightly bound to the nucleus. For example, in silicon, the 1s and 2p electrons have kinetic energies of hundreds of Rydbergs. Their wavefunctions oscillate rapidly near the nucleus—you need plane-wave cutoffs $E_{\mathrm{cut}} \sim 10\text{–}100$ Ry to represent them. Valence electrons oscillate more gently: for Si, the 3s and 3p electrons need $E_{\mathrm{cut}} \sim 30$ Ry.

The key insight is that core electrons are **spectators**: they do not participate in chemical bonding. What matters is the potential they create and its effect on valence electrons.

**The pseudopotential idea:** Replace the Coulomb potential of the nucleus minus the electron-density repulsion of the core with a softer, effective ionic potential $V_{\mathrm{ps}}(\mathbf{r})$. This potential is constructed so that:

1. **Pseudo-wavefunctions** $|\phi_{\mathrm{ps}}\rangle$ (smooth, nodeless) give the same scattering phase shifts as the all-electron wavefunctions.
2. Below the cutooff radius $r_c$ (typically 1–2 bohr), the pseudo-wavefunction is smooth.
3. Above $r_c$, the pseudo-wavefunction matches the all-electron wavefunction exactly.

The result: a plane-wave basis with 5–10× fewer components, and no loss of accuracy in properties like band structure, forces, or total energy.

## Norm-Conservation: The Hamann-Schlüter-Chiang Condition (1979)

**Norm-conservation** is a stability criterion that ensures transferability of the pseudopotential across different chemical environments and pressure ranges.

For each angular momentum channel $l$ and radial node $i$, the pseudo-wavefunction $\phi_{\mathrm{ps},i}^l(r)$ must satisfy:

$$\int_0^{r_c} |\phi_{\mathrm{ps},i}^l(r)|^2 r^2 \, dr = \int_0^{r_c} |\phi_{\mathrm{ae},i}^l(r)|^2 r^2 \, dr$$

**Why this matters:** The scattering phase shift $\delta_l(E)$ is related to the derivative of the logarithmic radial wavefunction at $r_c$. When norm is conserved, the first derivative of the phase shift with respect to energy,

$$\frac{d\delta_l}{dE}\bigg|_{E_{\mathrm{ref}}} \approx 0,$$

is minimized. This means the phase shift changes slowly near the reference energy $E_{\mathrm{ref}}$ used to construct the pseudopotential. A pseudopotential is then **transferable**: it works across a range of densities, pressures, and bonding environments with high accuracy. The penalty is a slightly softer potential (smaller $E_{\mathrm{cut}}$) compared to harder, less-transferable pseudopotentials.

## Kleinman-Bylander Separable Form (1982)

Evaluating the nonlocal potential naively,

$$\langle \mathbf{r} | \hat{V}_{\mathrm{NL}} | \psi \rangle = \sum_{\mathbf{R}} \int d^3r' \, V_{\mathrm{NL}}(\mathbf{r}, \mathbf{r}') \psi(\mathbf{r}'),$$

requires applying a dense nonlocal operator to every wavefunction—cost $O(N_{\mathrm{PW}}^2)$ per k-point, which is prohibitive.

The **Kleinman-Bylander** decomposition (1982) expresses the nonlocal part as a sum of rank-1 projectors:

$$\hat{V}_{\mathrm{NL}} = \sum_{a} \sum_{lm} \sum_{i,j} |\beta_{lm,i}^a\rangle D_{ij}^{l,a} \langle \beta_{lm,j}^a|,$$

where:
- $a$ indexes atoms
- $l, m$ are angular momentum quantum numbers
- $i, j$ index radial basis functions for channel $l$ (typically $i, j \in \{1, 2, \ldots, 4\}$)
- $\beta_{lm,i}^a(\mathbf{r}) = R_i^l(r) Y_{lm}(\hat{\mathbf{r}}) \, \delta(\mathbf{r} - \mathbf{R}_a)$ is a projector (smooth, localized at atom $a$)
- $D_{ij}^{l,a}$ is a small matrix (typically 4×4)

The action on a wavefunction becomes:

$$\hat{V}_{\mathrm{NL}} |\psi_\mathbf{k} \rangle = \sum_{a} \sum_{lm} \sum_{i,j} \left| \beta_{lm,i}^a \right\rangle D_{ij}^{l,a} \left( \int d^3r \, \beta_{lm,j}^a(\mathbf{r})^* \psi_\mathbf{k}(\mathbf{r}) \right),$$

which costs $O(N_{\mathrm{atoms}} \times N_{\mathrm{proj}} \times N_{\mathrm{PW}})$—typically 1000× faster.

**Geometric interpretation:** The projectors $\beta_{lm,i}^a$ are fixed basis functions. The matrix elements $D_{ij}^l$ are fitted so that this separable form reproduces the full nonlocal potential to high accuracy. For NC-PP constructed via Kleinman-Bylander, the fit is near-perfect; some modern hardened potentials (ONCV) may use slightly larger errors in the fit to gain transferability.

## Nonlocal Projectors in Reciprocal Space

To apply projectors efficiently on the plane-wave grid, KRONOS Fourier-transforms them. For atom $a$ at position $\mathbf{R}_a$, the Fourier components are:

$$\tilde{\beta}_{lm,i}^a(\mathbf{G}) = e^{-i\mathbf{G} \cdot \mathbf{R}_a} \int d^3r \, R_i^l(r) Y_{lm}(\hat{\mathbf{r}}) e^{-i\mathbf{G} \cdot \mathbf{r}}.$$

The radial integral is:

$$\int_0^\infty R_i^l(r) j_l(Gr) r^2 \, dr,$$

where $j_l$ is a spherical Bessel function. This is computed by numerical integration on the radial mesh from the UPF file and cached per k-point.

The projection of a wavefunction,

$$p_{lm,i}^a = \langle \beta_{lm,i}^a | \psi_\mathbf{k} \rangle = \sum_\mathbf{G} \tilde{\beta}_{lm,i}^a(\mathbf{G})^* \, \psi_\mathbf{k}(\mathbf{G}),$$

is a simple inner product. Applying the nonlocal operator requires only BLAS operations on the projection coefficients and the $D$ matrix.

**Memory note:** For a typical atom, 8–20 projectors ($l=0, 1, 2, 3$, each with $i \in \{1, 2\}$ or more) and hundreds of k-points, caching per k-point adds a few MB per atom—easily manageable.

## Local Pseudopotential: Coulomb Tail Subtraction

The **local part** of the pseudopotential, $V_{\mathrm{loc}}(\mathbf{r})$, is the spherically symmetric potential that includes the Coulomb repulsion from the nucleus minus the core density:

$$V_{\mathrm{loc}}(r) = -\frac{Z_{\mathrm{ion}}}{r} + \int d^3r' \, \frac{\rho_{\mathrm{core}}(r')}{|\mathbf{r} - \mathbf{r}'|}.$$

At large $r$, this tail diverges as $-Z/r$. In Rydberg units, the Fourier transform of the long-range Coulomb tail $-Z/r$ is:

$$\mathcal{F}\left[ -\frac{Z}{r} \right] = -\frac{8\pi Z}{G^2}.$$

**The problem:** The short-range remainder, $V_{\mathrm{loc}}(r) + Z/r$, is smooth and decays rapidly. Numerically tabulating it and Fourier-transforming is clean. But if you include the full long-range tail in the numerical FFT, you accumulate errors at small $G$ (the slowly-decaying Fourier components).

**The solution:** KRONOS subtracts the long-range tail analytically:

1. Numerically Fourier-transform only the short-range part: $\tilde{V}_{\mathrm{short}}(\mathbf{G}) = \mathcal{F}[V_{\mathrm{loc}} + Z/r]$.
2. Add back the analytic Coulomb FT: $\tilde{V}_{\mathrm{loc}}(\mathbf{G}) = \tilde{V}_{\mathrm{short}}(\mathbf{G}) - 8\pi Z / G^2$ (for $G > 0$).
3. At $G = 0$ (the mean-field energy), use $\int V_{\mathrm{loc}}(r) r^2 dr$ directly.

This "Coulomb tail subtraction" is numerically stable and avoids long-range errors in the potential.

## The Unified Pseudopotential Format (UPF v2)

The **UPF format** is a quasi-XML standard for storing pseudopotentials. KRONOS parses UPF v2 in `src/io/upf_parser.cpp`. A typical UPF file contains:

- **`<PP_HEADER>`**: Element, Z, pseudopotential type (NC-PP, PAW, Ultra-soft), functional (LDA, GGA), reference pressure/configuration.
- **`<PP_R>`**: Radial mesh with $N_r \sim 500$–1000 points, logarithmic spacing.
- **`<PP_LOCAL>`**: Local pseudopotential $V_{\mathrm{loc}}(r)$ on the radial mesh.
- **`<PP_NONLOCAL>`**: Nonlocal projectors.
  - **`<PP_BETA>`**: Radial projector functions $R_i^l(r)$, indexed by $l$, $i$.
  - **`<PP_DIJ>`**: The $D_{ij}^l$ matrices.
- **`<PP_PSWFC>`**: Pseudo-wavefunction $\phi_{\mathrm{ps},i}^l(r)$ for each channel (for testing/verification).
- **`<PP_AUGMENTATION>`** (PAW only): Augmentation charge $Q_{ij}^l(r)$, one-center overlap, compensation charges.

**Parsing and validation:** When KRONOS loads a UPF file, it:

1. Checks that the pseudo-wavefunctions are nodeless inside $r_c$.
2. For NC-PP, verifies norm-conservation to $\sim 10^{-5}$ for each channel.
3. Caches the radial functions and interpolates onto the FFT grid as needed.

This validation catches malformed or accidentally-swapped pseudopotentials early.

## PAW Augmentation (Blöchl 1994)

Norm-conserving pseudopotentials are robust and transferable, but they sacrifice some accuracy in the core region. The **Projector-Augmented Wave** (PAW) method reconstructs the all-electron wavefunction inside the core, achieving AE-level accuracy without requiring large basis sets.

The PAW ansatz partitions the wavefunction:

$$|\psi\rangle = |\tilde{\psi}\rangle + \sum_{a,ij} \left( |\phi_{ij}^a\rangle - |\tilde{\phi}_{ij}^a\rangle \right) \langle \tilde{p}_{ij}^a | \tilde{\psi} \rangle,$$

where:
- $|\tilde{\psi}\rangle$ is the pseudo-wavefunction (smooth, easily expanded in plane waves).
- $|\phi_{ij}^a\rangle$ are all-electron partial waves (frozen, precomputed).
- $|\tilde{\phi}_{ij}^a\rangle$ are pseudo partial waves (smooth matching versions).
- $|\tilde{p}_{ij}^a\rangle$ are projectors.

**Augmentation charge:** The charge density includes an "augmentation charge" $Q_{ij}^l(r)$ that corrects for the difference between the reconstructed core and the smooth pseudo density. This is where the extra accuracy comes from.

**In KRONOS:** The `PAWCalculator` class (`src/potential/paw.cpp`) handles:
- One-center overlap: overlap of AE partial waves.
- One-center energy: the frozen AE energy from the reference configuration.
- Augmentation energy: correction from $Q_{ij}$.

The main cost is recomputing these corrections in each SCF iteration—still $O(N_{\mathrm{atoms}})$ and negligible compared to FFT.

**Cutoff requirement:** PAW needs a finer FFT grid because augmentation charge varies faster inside the core. KRONOS enforces $E_{\mathrm{cut,rhoe}} \geq 12 \times E_{\mathrm{cut,wf}}$ for PAW (vs. $\geq 4 \times$ for NC-PP).

## Choosing a Pseudopotential: Soft vs. Hard

**Softness** is a trade-off:

| Characteristic | Soft PP | Hard PP |
|---|---|---|
| $E_{\mathrm{cut}}$ | 20–40 Ry | 60–150 Ry |
| Computational cost | Lower | Higher |
| Transferability | Excellent | Good–Excellent |
| Scattering region | Large $r_c$ | Compact |
| Fit difficulty | Easier | Harder |

**Element-dependent:**
- **Alkali metals** (Na, K): soft—valence is far from nucleus, core is well-separated.
- **3d Transition metals** (Fe, Ni, Cu): hard—semicore $3p$ electrons sit between valence $3d$ and core $1s$–$2p$; including $3p$ as core makes PP harder, but leaving it as valence explodes cutoff.
- **Oxygen, Nitrogen**: moderate—2s, 2p lie reasonably close.

**Pseudopotential libraries:**
- **PSlibrary**: QE standard; curated, tested across many systems.
- **SSSP-efficiency / SSSP-precision** (CCP NC): hand-optimized for high efficiency or maximum accuracy.
- **ONCV** (Schlipf-Gygi 2015): fully optimized Kleinman-Bylander projectors; excellent transferability; available for most elements.

KRONOS does not judge—whatever's in the UPF file will be evaluated correctly.

## How KRONOS Loads and Applies Pseudopotentials

**Initialization (at startup):**

1. `PseudoPotential` struct parses the UPF file via `src/io/upf_parser.cpp`.
2. `LocalPPEvaluator::compute_vlocal_g()` (in `src/potential/local_pp.cpp`):
   - Subtracts Coulomb tail analytically.
   - Numerically integrates the short-range part.
   - Caches $\tilde{V}_{\mathrm{loc}}(\mathbf{G})$ for all $\mathbf{G}$ in the basis.
3. `NonlocalPP::prepare_kpoint()` (in `src/potential/nonlocal_pp.cpp`):
   - For each atom and each projector, computes $\tilde{\beta}_{lm,i}^a(\mathbf{k}+\mathbf{G})$ via spherical Bessel transform.
   - Caches the Fourier components.

**Per SCF iteration:**

- **Local contribution:** $V_{\mathrm{loc}} \to$ real-space grid via IFFT, pointwise multiply with density/wavefunctions, FFT back. Cost: $O(N_{\mathrm{FFT}} \log N_{\mathrm{FFT}})$.
- **Nonlocal contribution:** Project wavefunctions onto $\beta$ via inner products, apply $D$ matrix, add back. Cost: $O(N_{\mathrm{atoms}} N_{\mathrm{proj}} N_{\mathrm{PW}})$.
- **PAW (if enabled):** `PAWCalculator::compute_one_center()` adds augmentation energy and forces.

The modular design means swapping NCPP ↔ PAW is just a configuration flag in `crystal.nonlocal_type`.

## References

- Hamann, D. R., Schlüter, M., Chiang, C. "Norm-Conserving Pseudopotentials." *Phys. Rev. Lett.* **43**, 1494 (1979).
- Kleinman, L. & Bylander, D. M. "Efficacious Form for Model Pseudopotentials." *Phys. Rev. Lett.* **48**, 1425 (1982).
- Blöchl, P. E. "Projector Augmented-Wave Method." *Phys. Rev. B* **50**, 17953 (1994).
- Hamann, D. R. "Optimized Norm-Conserving Pseudopotentials." *Phys. Rev. B* **88**, 085117 (2013).
- Schlipf, M. & Gygi, F. "Optimization Algorithm for the Generation of ONCV Pseudopotentials." *Comput. Phys. Commun.* **196**, 36 (2015).
- Martin, R. M. *Electronic Structure: Basic Theory and Practical Methods*. Ch. 11, Cambridge University Press (2004).
- Pickett, W. E. "Pseudopotential Methods in Condensed Matter Applications." *Comput. Phys. Rep.* **9**, 115 (1989).
