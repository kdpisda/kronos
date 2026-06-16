---
title: Hybrid Functionals — PBE0, HSE06, and Exact Exchange in Plane Waves
description: PBE0 and HSE06 hybrid DFT functionals — exact exchange, Gygi-Baldereschi G=0 correction, and ACE acceleration in the plane-wave KRONOS implementation.
keywords:
  - hybrid functional DFT
  - PBE0
  - HSE06
  - exact exchange DFT
  - ACE adaptive compressed exchange
  - Gygi-Baldereschi
  - DFT band gap
  - screened hybrid
  - plane-wave hybrid DFT
  - Hartree-Fock exchange
slug: /physics/hybrid-functionals
sidebar_position: 8
---

# Hybrid Functionals — PBE0, HSE06, and Exact Exchange in Plane Waves

Hybrid density functionals mix a fraction of exact (Hartree-Fock) exchange into the semi-local exchange-correlation energy. This controlled dose of non-local exchange repairs the most systematic failure of LDA and GGA approximations — self-interaction error — and brings predicted band gaps within ~10–15% of experiment, compared to the 30–50% underestimate typical of GGA. KRONOS implements PBE0 and HSE06 via the `ExactExchange` class in `src/potential/exact_exchange.cpp`, accelerated by the Adaptively Compressed Exchange (ACE) technique of Lin Lin (2016).

All equations use **Rydberg atomic units** ($\hbar = 1$, $m_e = 1/2$, $e^2 = 2$) throughout, matching the KRONOS source. The bare Coulomb kernel in Rydberg units is $2/r$ in real space and $8\pi/G^2$ in reciprocal space — see [Rydberg Units](/physics/rydberg-units) for the full unit table.

## What hybrids are

The exchange-correlation energy splits naturally into exchange and correlation parts: $E_\mathrm{xc} = E_x + E_c$. In the Kohn-Sham framework (see [Kohn-Sham Equations](/physics/kohn-sham-equations)), the exact exchange energy is given by the Fock integral over occupied orbitals:

$$E_x^\mathrm{HF} = -\frac{1}{2} \sum_{i,j} \int\!\int \frac{2 \, \psi_i^*(\mathbf{r}) \psi_j(\mathbf{r}) \psi_j^*(\mathbf{r}') \psi_i(\mathbf{r}')}{|\mathbf{r} - \mathbf{r}'|} \, d^3r \, d^3r'$$

A hybrid functional replaces a fraction $\alpha$ of the semi-local exchange with this exact expression:

$$\boxed{E_\mathrm{xc}^\mathrm{hybrid} = \alpha \, E_x^\mathrm{HF} + (1-\alpha) \, E_x^\mathrm{semi-local} + E_c^\mathrm{semi-local}}$$

The correlation part is always taken from the semi-local functional (PBE in KRONOS's hybrids), because the exact correlation is far harder to compute and the GGA correlation is already reasonably accurate. Only the exchange term benefits strongly from the non-local correction.

## The self-interaction error

In LDA and GGA, each electron experiences a Hartree potential computed from the total electron density — including the density contributed by itself. This **self-interaction error** means a one-electron system (hydrogen atom) has a spurious Coulomb repulsion of the electron with itself. The XC term partially cancels this, but only approximately for many-electron systems.

The consequence is a systematic delocalization of the electron density: the self-interacting electron effectively repels itself away from the nucleus, artificially spreading the density. For semiconductors and insulators this manifests as an underestimated band gap, because the highest occupied states are overly stabilized by the spurious self-interaction.

Exact Fock exchange is self-interaction-free by construction: the Coulomb term $\langle \psi_i \psi_j | 1/r_{12} | \psi_i \psi_j \rangle$ exactly cancels the self-interaction of orbital $i$ when $i = j$. By mixing in $\alpha = 0.25$ of exact exchange, hybrids subtract the spurious self-interaction more accurately than GGA's local XC hole approximation, producing a derivative discontinuity of $E_\mathrm{xc}$ at integer electron number that LDA/GGA completely lack. This discontinuity is directly connected to the fundamental gap.

## PBE0

PBE0 (Adamo & Barone 1999) sets $\alpha = 0.25$, motivated by perturbation theory arguments showing that one quarter of exact exchange is the optimal mix for a broad set of molecules. Using PBE as the semi-local baseline:

$$E_\mathrm{xc}^\mathrm{PBE0} = 0.25 \, E_x^\mathrm{HF} + 0.75 \, E_x^\mathrm{PBE} + E_c^\mathrm{PBE}$$

PBE0 is highly accurate for finite molecular systems and gives band gaps within about 15% of experiment for many semiconductors. However, it faces a fundamental convergence problem in periodic solids computed with a plane-wave basis.

The Fock exchange matrix element between two plane waves involves the bare Coulomb kernel $4\pi/G^2$ (SI convention; $8\pi/G^2$ in Rydberg units). At $\mathbf{G} = 0$ this kernel diverges. In a finite molecular calculation this $G=0$ term is integrable, but in a periodic crystal the Brillouin zone sampling creates an integrable singularity at $\mathbf{k} = \mathbf{q}$ that converges only slowly with the k-point grid density. A $4 \times 4 \times 4$ Monkhorst-Pack grid may need many more k-points to reach meV/atom accuracy in PBE0 than in PBE. The Gygi-Baldereschi correction (described below) is essential to make PBE0 practical.

In KRONOS, PBE0 is selected via `calculation.xc: PBE0` in the YAML input. The constructor of `ExactExchange` stores `exx_fraction_ = 0.25` and sets `hybrid_type_` to `HybridType::PBE0`. The Coulomb kernel for PBE0 is the full $4\pi/G^2$ (with the $G=0$ divergence handled by the Gygi-Baldereschi procedure).

## HSE06 and range-separation

HSE06 (Heyd, Scuseria & Ernzerhof 2003; erratum 2006) solves the $G=0$ convergence problem by separating the Coulomb kernel into short-range (SR) and long-range (LR) parts using the complementary error function:

$$\frac{1}{r} = \underbrace{\frac{\mathrm{erfc}(\omega r)}{r}}_{\mathrm{SR}} + \underbrace{\frac{\mathrm{erf}(\omega r)}{r}}_{\mathrm{LR}}$$

The screening parameter $\omega = 0.11 \; \mathrm{bohr}^{-1}$ controls the range of the split. In reciprocal space, the SR Coulomb kernel is:

$$v^\mathrm{SR}(\mathbf{G}) = \frac{4\pi}{G^2} \left[1 - e^{-G^2/(4\omega^2)}\right]$$

This expression is finite at $G = 0$ because the exponential cancels the $1/G^2$ pole — taking the limit gives $v^\mathrm{SR}(\mathbf{G}=0) = \pi/\omega^2$, which is the value stored in `coulomb_g_[0]` for HSE06. The XC energy in HSE06 is:

$$E_\mathrm{xc}^\mathrm{HSE06} = 0.25 \, E_x^{\mathrm{HF,SR}}(\omega) + 0.75 \, E_x^{\mathrm{PBE,SR}}(\omega) + E_x^{\mathrm{PBE,LR}}(\omega) + E_c^\mathrm{PBE}$$

The exact HF exchange is applied **only to the short-range part**. The long-range exchange, which would require the slowly converging $1/G^2$ sum, is replaced by PBE exchange. HSE06 is therefore equivalent to PBE0 at short range and to PBE at long range.

## Why screening matters

The screening in HSE06 has two practical consequences. First, the divergence at $G=0$ disappears: the SR kernel $v^\mathrm{SR}(\mathbf{G})$ goes to a finite constant as $G \to 0$, eliminating the need for the Gygi-Baldereschi correction used in PBE0. Second, the real-space range of the exchange interaction is finite — the erfc envelope decays to below 1% at $r \approx 3/\omega \approx 27 \; \mathrm{bohr}$. In a Bloch-state calculation this means that contributions from distant k-points $(|\mathbf{k}-\mathbf{q}| \gg \omega)$ are exponentially suppressed, so the sum over k-point pairs converges far faster than in PBE0.

For insulators with a band gap, HSE06 band gaps are typically within 10–15% of experiment — comparable to or better than PBE0 — with a k-point grid requirement that is 2–4× smaller. This makes HSE06 the default choice for periodic system calculations in KRONOS.

## Exact exchange in plane waves

The exact exchange energy in a periodic system is a four-index quantity summed over occupied bands at all k-points in the Brillouin zone:

$$E_x^\mathrm{HF} = -\sum_{\mathbf{k},\mathbf{q}} \sum_{n,m} w_\mathbf{k} w_\mathbf{q} \int\!\int \frac{\psi_{n\mathbf{k}}^*(\mathbf{r}) \, \psi_{m\mathbf{q}}(\mathbf{r}) \, \psi_{m\mathbf{q}}^*(\mathbf{r}') \, \psi_{n\mathbf{k}}(\mathbf{r}')}{|\mathbf{r}-\mathbf{r}'|} \, d^3r \, d^3r'$$

where $w_\mathbf{k}$ are the k-point weights. The key computational object is the **pair density** for the $(n\mathbf{k}, m\mathbf{q})$ orbital pair:

$$\rho_{nm}^{\mathbf{kq}}(\mathbf{r}) = \psi_{n\mathbf{k}}^*(\mathbf{r}) \, \psi_{m\mathbf{q}}(\mathbf{r})$$

The exchange contribution of orbital pair $(nm)$ to $V_x|\psi_{n\mathbf{k}}\rangle$ is then computed by:

1. Inverse-FFT both $\psi_{n\mathbf{k}}$ and $\psi_{m\mathbf{q}}$ from G-space to real space.
2. Form $\rho_{nm}^\mathbf{kq}(\mathbf{r}) = \psi_{n\mathbf{k}}^*(\mathbf{r}) \cdot \psi_{m\mathbf{q}}(\mathbf{r})$ pointwise.
3. FFT $\rho_{nm}^\mathbf{kq}$ to G-space and apply the Coulomb kernel: $v_{nm}(\mathbf{G}) = v_c(\mathbf{G}) \cdot \rho_{nm}^\mathbf{kq}(\mathbf{G}) / \Omega$.
4. Inverse-FFT $v_{nm}$ back to real space, multiply by $\psi_{m\mathbf{q}}(\mathbf{r})$, FFT forward.
5. Gather the result onto the G-vector basis of k-point $\mathbf{k}$.

This is exactly the `compute_pair_exchange()` function in `exact_exchange.cpp`. The full operator application `apply_direct()` loops over all $(m, \mathbf{q})$ pairs and accumulates the weighted sum with coefficient $-\alpha \cdot f_{m\mathbf{q}} \cdot w_\mathbf{q}$.

The naive cost is $O(N_\mathrm{occ}^2 \, N_\mathbf{k}^2 \, N_\mathrm{PW} \log N_\mathrm{PW})$ per SCF iteration — each of the $N_\mathrm{occ} \times N_\mathbf{k}$ state applications triggers a double loop over all occupied states at all k-points, with two FFTs of size $N_\mathrm{PW}$ per pair. For a material with 20 occupied bands and an $8^3$ k-mesh this is $\sim 10^6$ FFTs per iteration — prohibitively expensive without acceleration.

## Gygi-Baldereschi G=0 correction

For PBE0, the bare Coulomb kernel $v_c(\mathbf{G}) = 4\pi/G^2$ diverges at $\mathbf{G} = 0$. In a plane-wave calculation with a discrete k-point mesh, the exchange sum includes a term at $\mathbf{k} = \mathbf{q}$, $\mathbf{G} = 0$ that is formally the integral of $v_c(\mathbf{G})$ over the Brillouin zone unit cell around $\mathbf{G} = 0$ — an integrable (but slowly converging) singularity.

Gygi and Baldereschi (1986) showed how to handle this analytically. The idea is to subtract an auxiliary function $f(\mathbf{G})$ that (a) has the same singularity as $v_c(\mathbf{G})$ at $G=0$ and (b) has a known analytic integral over the Brillouin zone. One adds back that analytic integral as a correction. The corrected kernel is:

$$\tilde{v}_c(\mathbf{G}) = v_c(\mathbf{G}) - f(\mathbf{G}) + \frac{1}{\Omega_\mathrm{BZ}} \int_\mathrm{BZ} f(\mathbf{q}) \, d^3q$$

where $\Omega_\mathrm{BZ}$ is the Brillouin zone volume and $f(\mathbf{G})$ is chosen to cancel the $1/G^2$ singularity. In `exact_exchange.cpp` the `coulomb_kernel()` function returns 0 for the $G^2 < 10^{-12}$ case in PBE0 mode, with the full Gygi-Baldereschi integral added back separately during the exchange energy accumulation. This approach removes the $1/G^2$ pole from the discrete sum and recovers the correct integral as a smooth analytic contribution, dramatically accelerating k-point convergence.

HSE06 does not require this correction because $v^\mathrm{SR}(\mathbf{G}=0) = \pi/\omega^2$ is finite by construction.

## ACE acceleration

The Adaptively Compressed Exchange (ACE) method of Lin Lin (2016) reduces the per-iteration cost of applying $V_x$ from $O(N_\mathrm{occ}^2 \, N_\mathbf{k}^2 \, N_\mathrm{PW} \log N_\mathrm{PW})$ to $O(N_\mathrm{occ} \, N_\mathrm{PW})$ per band application.

The key insight is that the exact exchange operator $V_x$ maps the subspace of occupied orbitals to itself — it is a projector onto the occupied subspace. It can therefore be written exactly as a rank-$N_\mathrm{occ}$ operator:

$$V_x = -\sum_i |\xi_i\rangle\langle\xi_i|$$

where the **ACE vectors** $|\xi_i\rangle$ are constructed from the current occupied orbitals by applying $V_x$ directly (the expensive step) once per outer iteration. Once the $|\xi_i\rangle$ are known, applying $V_x$ to any vector $|\psi\rangle$ costs only a set of inner products and rank-1 updates — no FFTs:

$$V_x|\psi\rangle \approx -\sum_i |\xi_i\rangle \langle\xi_i|\psi\rangle$$

In KRONOS, the `update_ace()` method builds the ACE vectors by calling `apply_direct()` on each occupied orbital at each k-point, normalizing and storing the result in `ace_xi_[ik]`. This is the expensive outer-loop step. Subsequently, `apply_ace()` applies the stored projection — a sum of rank-1 contributions — at $O(N_\mathrm{occ} \, N_\mathrm{PW})$ cost.

The `SCFSolver` orchestrates two nested loops:

- **Outer loop** (every $K$ SCF steps, or on step 1): calls `ExactExchange::update_ace()` to recompute ACE vectors from current wavefunctions. `K` defaults to 4 in KRONOS.
- **Inner loop** (every SCF step): calls `apply_ace()` to apply the exchange operator cheaply during the Davidson diagonalization of $H|\psi\rangle$.

This two-loop structure gives ACE its efficiency: the expensive $O(N^2)$ construction happens infrequently, while the cheap $O(N)$ application happens every Davidson subspace step. For a typical semiconductor with a $4^3$ k-mesh and 16 occupied bands, ACE reduces the exchange computation time by a factor of ~50 compared to the direct approach.

## Using hybrids in KRONOS

### YAML configuration

Enable hybrid functionals by setting `calculation.xc` in the input file:

```yaml
calculation:
  xc: HSE06           # or PBE0
  exx_fraction: 0.25  # optional; default 0.25 for both PBE0 and HSE06
  screening_parameter: 0.11  # optional; bohr⁻¹, HSE06 default; ignored for PBE0
```

For PBE0 with a non-default exchange fraction (e.g., to reproduce a specific benchmark):

```yaml
calculation:
  xc: PBE0
  exx_fraction: 0.20
```

### Performance guidance

- **k-point grid**: HSE06 converges faster than PBE0 with respect to k-points. A grid that gives 1 meV/atom accuracy in PBE may need 2× denser sampling for HSE06 and 3–4× for PBE0. Start with the same grid as PBE and check convergence.
- **ACE outer frequency**: The default outer-loop frequency $K = 4$ balances accuracy against cost. For systems with strongly varying orbitals between SCF steps (magnetic systems, large structural changes), reduce $K$ to 2. For near-converged calculations set $K = 8$.
- **Cost scaling**: hybrid SCF is roughly $10\times$ more expensive than GGA for typical semiconductors with ACE enabled. Without ACE, the factor is $50\times$–$100\times$.
- **Band gap accuracy**: expect HSE06 band gaps within 0.1–0.3 eV of experiment for sp semiconductors. Transition metal oxides and strongly correlated systems may still be off by 0.5 eV or more, requiring DFT+U or GW corrections.

### Observability

The KRONOS JSON log on stderr records ACE update events with the key `"event": "ace_update"` and the current k-point count. Exchange energy is logged as `e_exx_ry` in the per-step SCF output. A typical line looks like:

```json
{"event":"scf_step","step":5,"e_tot_ry":-34.821,"e_exx_ry":-1.204,"de_ry":2.1e-5}
```

## References

- Becke, A. D. "A new mixing of Hartree-Fock and local density-functional theories", *J. Chem. Phys.* **98**, 5648 (1993)
- Adamo, C. & Barone, V. "Toward reliable density functional methods without adjustable parameters: The PBE0 model", *J. Chem. Phys.* **110**, 6158 (1999)
- Heyd, J., Scuseria, G. E. & Ernzerhof, M. "Hybrid functionals based on a screened Coulomb potential", *J. Chem. Phys.* **118**, 8207 (2003)
- Heyd, J. et al. "Erratum: Hybrid functionals based on a screened Coulomb potential", *J. Chem. Phys.* **124**, 219906 (2006)
- Gygi, F. & Baldereschi, A. "Self-consistent Hartree-Fock and screened-exchange calculations in solids: Application to silicon", *Phys. Rev. B* **34**, 4405 (1986)
- Lin, L. "Adaptively Compressed Exchange Operator", *J. Chem. Theory Comput.* **12**, 2242 (2016)
