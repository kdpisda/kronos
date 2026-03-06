# KRONOS Physics Notes

> All equations below use **Rydberg atomic units** ($\hbar = 1$, $m_e = 1/2$, $e^2 = 2$, $a_0 = 1$), matching the KRONOS source code.  Key consequences: kinetic prefactor is 1 (not $1/2$), Hartree prefactor is $8\pi$ (not $4\pi$), bare Coulomb potential is $-2Z/r$ (not $-Z/r$).

---

## 1. Kohn-Sham DFT Framework

### 1.1 Hohenberg-Kohn theorems

Density Functional Theory rests on two theorems by Hohenberg and Kohn (1964):

1. The ground-state total energy of an interacting electron system is a unique functional of the electron density $n(\mathbf{r})$.
2. The true ground-state density minimizes this functional.

### 1.2 Kohn-Sham equations

Kohn and Sham (1965) reformulated the problem as a set of single-particle equations whose orbitals $\psi_{n\mathbf{k}}$ reproduce the exact density:

$$n(\mathbf{r}) = \sum_{n\mathbf{k}} f_{n\mathbf{k}} \, |\psi_{n\mathbf{k}}(\mathbf{r})|^2$$

where $f_{n\mathbf{k}}$ are occupation numbers (including k-point weight and spin degeneracy).

The Kohn-Sham equation in Rydberg units reads:

$$\bigl[-\nabla^2 + V_\mathrm{eff}(\mathbf{r})\bigr] \psi_{n\mathbf{k}}(\mathbf{r}) = \varepsilon_{n\mathbf{k}} \, \psi_{n\mathbf{k}}(\mathbf{r})$$

where the effective potential is:

$$V_\mathrm{eff}(\mathbf{r}) = V_H(\mathbf{r}) + V_{xc}(\mathbf{r}) + V_\mathrm{loc}(\mathbf{r})$$

plus the nonlocal pseudopotential operator $\hat{V}_\mathrm{NL}$ acting in reciprocal space.  Since $V_\mathrm{eff}$ depends on $n$, which depends on the orbitals, which depend on $V_\mathrm{eff}$, the equations must be solved self-consistently (the SCF loop).

### 1.3 Total energy functional

The total energy in Rydberg units is:

$$E_\mathrm{tot} = E_\mathrm{band} - E_H + E_{xc} - \int V_{xc}(\mathbf{r}) \, n(\mathbf{r}) \, d\mathbf{r} + E_\mathrm{Ewald}$$

where $E_\mathrm{band} = \sum_{n\mathbf{k}} f_{n\mathbf{k}} \, \varepsilon_{n\mathbf{k}}$ is the sum of eigenvalues.  The double-counting corrections ($-E_H$ and $-\int V_{xc} \, n$) remove terms counted twice in $E_\mathrm{band}$.  The Ewald energy $E_\mathrm{Ewald}$ is the classical ion-ion interaction.

In code (`scf.cpp`), the band energy decomposition is:

$$E_\mathrm{band} = E_\mathrm{kin} + 2 E_H + E_\mathrm{loc} + E_\mathrm{NL} + \int V_{xc} \, n$$

from which the kinetic energy is extracted as:

$$E_\mathrm{kin} = E_\mathrm{band} - 2 E_H - E_\mathrm{loc} - E_\mathrm{NL} - \int V_{xc} \, n$$

For metallic systems with smearing, the free energy includes an entropy term $-TS$ that regularizes partial occupations near the Fermi level.

---

## 2. Plane-Wave Formalism

### 2.1 Bloch theorem and plane-wave expansion

In a periodic crystal, Bloch's theorem states:

$$\psi_{n\mathbf{k}}(\mathbf{r}) = e^{i\mathbf{k}\cdot\mathbf{r}} \, u_{n\mathbf{k}}(\mathbf{r})$$

where $u_{n\mathbf{k}}$ has the periodicity of the lattice.  Expanding $u_{n\mathbf{k}}$ in reciprocal lattice vectors $\mathbf{G}$:

$$\psi_{n\mathbf{k}}(\mathbf{r}) = \sum_{\mathbf{G}} c_{n\mathbf{k}}(\mathbf{G}) \, e^{i(\mathbf{k}+\mathbf{G})\cdot\mathbf{r}}$$

KRONOS stores and manipulates the coefficients $c_{n\mathbf{k}}(\mathbf{G})$ as complex128 (double-precision complex) vectors.

### 2.2 Kinetic energy cutoff

The basis is truncated by the energy cutoff `ecutwfc` (in Ry):

$$|\mathbf{k}+\mathbf{G}|^2 \le E_\mathrm{cut}$$

Note the absence of the factor $1/2$ --- in Rydberg units the kinetic energy operator is $-\nabla^2$, so $T_\mathbf{G} = |\mathbf{k}+\mathbf{G}|^2$.

KRONOS uses a shared G-vector basis expanded to cover all k-points: the basis includes every $\mathbf{G}$ satisfying $|\mathbf{k}+\mathbf{G}|^2 \le E_\mathrm{cut}$ for *any* k-point in the irreducible Brillouin zone.  When applying $H|\psi\rangle$ at a specific k-point, G-vectors outside the per-k cutoff are masked to zero and assigned a high energy wall ($10^4$ Ry) so the Davidson solver drives their amplitudes to zero.

### 2.3 FFT dual representation

Plane waves make two operations diagonal in complementary spaces:

| Operation | Diagonal in | Cost |
|-----------|-------------|------|
| Kinetic energy $T\|\psi\rangle$ | G-space: $|\mathbf{k}+\mathbf{G}|^2 \, c(\mathbf{G})$ | $O(N_\mathrm{pw})$ |
| Local potential $V\|\psi\rangle$ | Real space: $V(\mathbf{r}) \, \psi(\mathbf{r})$ | $O(N_\mathrm{grid})$ |

Switching between representations costs $O(N \log N)$ via FFT.  The density cutoff grid satisfies `ecutrho` $\ge 4 \times$ `ecutwfc` for norm-conserving PPs, ensuring that the product $V(\mathbf{r})\psi(\mathbf{r})$ is alias-free.

### 2.4 K-point sampling

KRONOS generates Monkhorst-Pack k-point grids.  The k-point formula is:

$$k_i = \frac{2n_i - N_i - 1}{2N_i} + \frac{s_i}{2N_i}$$

where $n_i = 1, \ldots, N_i$, and $s_i \in \{0, 1\}$ is the grid shift.  Time-reversal symmetry ($\mathbf{k} \leftrightarrow -\mathbf{k}$) and, when spglib is available, full space-group symmetry reduce the grid to the irreducible Brillouin zone.

---

## 3. Pseudopotential Theory

### 3.1 Why pseudopotentials

Core electrons create rapid wavefunction oscillations near nuclei that demand enormous plane-wave cutoffs.  Pseudopotentials replace the true ionic potential plus core electrons with a smooth effective potential that reproduces the correct valence physics outside a cutoff radius $r_c$.

### 3.2 Norm-conserving condition

KRONOS v0.1 uses norm-conserving pseudopotentials, which satisfy:

$$\int_0^{r_c} |\tilde\phi_l(r)|^2 \, r^2 \, dr = \int_0^{r_c} |\phi_l(r)|^2 \, r^2 \, dr$$

This guarantees correct scattering properties and transferability.  KRONOS verifies this condition when loading UPF pseudopotential files.

### 3.3 Kleinman-Bylander separable form

The semilocal pseudopotential $V_\mathrm{PS} = V_\mathrm{loc}(r) + \sum_l |l\rangle \delta V_l \langle l|$ is separated into local and nonlocal parts.  The nonlocal part is written in the efficient Kleinman-Bylander separable form:

$$\hat{V}_\mathrm{NL} = \sum_{a,i,j} |\beta_i^a\rangle \, D_{ij}^a \, \langle\beta_j^a|$$

where $\beta_i^a$ are projector functions centered on atom $a$, and $D_{ij}^a$ is the coupling matrix (block-diagonal in angular momentum quantum numbers $l, m$).  Each UPF beta projector with angular momentum $l$ generates $2l+1$ expanded projectors indexed by $m = -l, \ldots, +l$.

### 3.4 Nonlocal projectors in reciprocal space

The projectors are evaluated in reciprocal space via spherical Bessel transforms:

$$\beta_{i,lm}^a(\mathbf{k}+\mathbf{G}) = \frac{4\pi}{\sqrt{\Omega}} \, i^l \, \int_0^\infty r \, \beta_i^\mathrm{UPF}(r) \, j_l(|\mathbf{k}+\mathbf{G}|r) \, dr \;\cdot\; Y_{lm}(\widehat{\mathbf{k}+\mathbf{G}}) \;\cdot\; e^{-i(\mathbf{k}+\mathbf{G})\cdot\boldsymbol{\tau}_a}$$

Note: UPF files store $r\beta(r)$, so the integrand is $r \cdot \beta^\mathrm{UPF}$, not $r^2 \cdot \beta$.  The radial integrals are evaluated by Simpson's rule on the UPF radial mesh, and spherical Bessel functions $j_l$ are computed analytically for $l \le 3$ with upward recurrence for $l \ge 4$.

Applying $\hat{V}_\mathrm{NL}|\psi\rangle$ requires:
1. Compute projections: $P_j^a = \langle\beta_j^a|\psi\rangle = \sum_\mathbf{G} \beta_j^{a*}(\mathbf{k}+\mathbf{G}) \, \psi(\mathbf{G})$
2. Apply $D$ matrix: $c_i^a = \sum_j D_{ij}^a \, P_j^a$
3. Expand: $(\hat{V}_\mathrm{NL}|\psi\rangle)_\mathbf{G} = \sum_{a,i} c_i^a \, \beta_i^a(\mathbf{k}+\mathbf{G})$

### 3.5 Coulomb tail subtraction for $V_\mathrm{loc}(G)$

The local potential $V_\mathrm{loc}(r) \to -2Z_v/r$ at large $r$ (Rydberg Coulomb tail), making the direct Fourier integral poorly convergent.  KRONOS splits:

$$V_\mathrm{loc}(r) = \underbrace{\bigl[V_\mathrm{loc}(r) + 2Z_v \operatorname{erf}(r/\sigma)/r\bigr]}_{V_\mathrm{short}(r),\ \text{numerical FT}} + \underbrace{\bigl[-2Z_v \operatorname{erf}(r/\sigma)/r\bigr]}_{V_\mathrm{long}(r),\ \text{analytic FT}}$$

The short-range part $V_\mathrm{short}(r) \to 0$ rapidly and its Fourier transform converges with a modest radial mesh.  The analytic Fourier transform of the long-range part is:

$$\widetilde{V}_\mathrm{long}(q) = -\frac{8\pi Z_v}{q^2} e^{-q^2\sigma^2/4}$$

At $q = 0$ the $1/q^2$ divergence cancels with the Hartree $G=0$ term (both set to zero) and the Ewald charged-cell correction.  The finite remainder kept is $+2\pi Z_v \sigma^2$.  KRONOS uses $\sigma = 1$ bohr.  The full form factor is:

$$V_\mathrm{loc}(q) = \frac{1}{\Omega}\left[4\pi \int_0^\infty r^2 V_\mathrm{short}(r) \frac{\sin(qr)}{qr} dr + \widetilde{V}_\mathrm{long}(q)\right]$$

---

## 4. Exchange-Correlation

### 4.1 LDA: Perdew-Zunger parametrization

In LDA the XC energy depends only on the local density:

$$E_{xc}^\mathrm{LDA}[n] = \int n(\mathbf{r}) \, \varepsilon_{xc}(n(\mathbf{r})) \, d\mathbf{r}$$

The XC potential entering the Kohn-Sham equations is the functional derivative:

$$V_{xc}(\mathbf{r}) = \frac{\delta E_{xc}}{\delta n} = \varepsilon_{xc}(n) + n \frac{d\varepsilon_{xc}}{dn}$$

**Exchange (Slater).** In Rydberg units, the exchange energy density per electron is:

$$\varepsilon_x(n) = -2 \left(\frac{3}{4\pi}\right)^{1/3} n^{1/3} = -\frac{3}{2} \left(\frac{3}{\pi}\right)^{1/3} n^{1/3}$$

with potential $V_x = (4/3)\varepsilon_x$.  The factor of 2 relative to Hartree units comes from the Ry conversion.

**Correlation (Perdew-Zunger 1981).** The Ceperley-Alder quantum Monte Carlo correlation energy for the homogeneous electron gas is parametrized in terms of $r_s = (3/4\pi n)^{1/3}$:

- For $r_s \ge 1$: $\varepsilon_c = \gamma / (1 + \beta_1 \sqrt{r_s} + \beta_2 r_s)$, with Rydberg parameters $\gamma = -0.2846$, $\beta_1 = 1.0529$, $\beta_2 = 0.3334$.
- For $r_s < 1$: $\varepsilon_c = A \ln r_s + B + C \, r_s \ln r_s + D \, r_s$, with Rydberg parameters $A = 0.0622$, $B = -0.096$, $C = 0.004$, $D = -0.0232$.

The correlation potential is $V_c = \varepsilon_c - (r_s/3) \, d\varepsilon_c/dr_s$.

When libxc is available, KRONOS delegates to it (with a factor of 2 for Ry conversion); otherwise it uses the built-in implementation above.

### 4.2 GGA: PBE functional

GGA functionals also depend on the density gradient:

$$E_{xc}^\mathrm{GGA}[n] = \int n(\mathbf{r}) \, \varepsilon_{xc}(n, |\nabla n|^2) \, d\mathbf{r}$$

The GGA potential has an additional correction beyond the LDA-like $\partial\varepsilon_{xc}/\partial n$ term:

$$V_{xc}^\mathrm{GGA} = \frac{\partial (n\varepsilon_{xc})}{\partial n} - 2\nabla\cdot\left[\frac{\partial(n\varepsilon_{xc})}{\partial(\nabla n)}\right]$$

In KRONOS, the gradient $\nabla n$ is computed in G-space ($i\mathbf{G}\,n(\mathbf{G})$) for each Cartesian direction, inverse-FFTed to real space, and the scalar $\sigma = |\nabla n|^2$ is formed pointwise.  The divergence correction $-2\nabla\cdot(v_\sigma \nabla n)$ is computed by forming $h_d(\mathbf{r}) = v_\sigma(\mathbf{r}) \cdot (\partial n / \partial x_d)(\mathbf{r})$, FFTing to G-space, multiplying by $i G_d$, and summing the three Cartesian components.

---

## 5. Hartree Potential

The Hartree (electron-electron Coulomb) potential in Rydberg units is diagonal in reciprocal space:

$$V_H(\mathbf{G}) = \frac{8\pi \, n(\mathbf{G})}{|\mathbf{G}|^2}$$

The prefactor $8\pi = 2 \times 4\pi$ accounts for the Rydberg unit convention.  The $\mathbf{G} = 0$ component is set to zero; this arbitrary constant cancels with the ionic $G=0$ terms and the Ewald correction.

The Hartree energy is:

$$E_H = \frac{\Omega}{2} \sum_\mathbf{G} V_H^*(\mathbf{G}) \, n(\mathbf{G})$$

In the KRONOS FFT convention, both $V_H$ and $n$ carry a factor of $N_\mathrm{grid}$ relative to the physics convention, so the energy formula includes a normalization by $N_\mathrm{grid}^2$.

---

## 6. Ewald Summation

### 6.1 The problem

The electrostatic energy of periodic point charges:

$$E_\mathrm{ion} = \frac{e^2}{2} \sideset{}{'}\sum_{i,j,\mathbf{T}} \frac{Z_i Z_j}{|\mathbf{r}_i - \mathbf{r}_j + \mathbf{T}|}$$

converges only conditionally (the $1/r$ tail extends over all lattice images).  In Rydberg units $e^2 = 2$.

### 6.2 Ewald splitting

Split $1/r$ into a short-range and a smooth long-range part:

$$\frac{1}{r} = \frac{\operatorname{erfc}(\eta r)}{r} + \frac{\operatorname{erf}(\eta r)}{r}$$

The parameter $\eta$ balances real- and reciprocal-space convergence.  KRONOS chooses $\eta = \sqrt{\pi}\,(N_\mathrm{atoms}/\Omega^2)^{1/6}$.

### 6.3 Four energy terms

**Real-space sum** (short-range, converges exponentially):

$$E_\mathrm{real} = \frac{e^2}{2} \sideset{}{'}\sum_{i,j,\mathbf{T}} Z_i Z_j \frac{\operatorname{erfc}(\eta\,|\mathbf{r}_{ij}+\mathbf{T}|)}{|\mathbf{r}_{ij}+\mathbf{T}|}$$

The real-space cutoff is $R_\mathrm{cut} = \max(6/\eta, \sqrt{-\ln\epsilon}/\eta)$ where $\epsilon = 10^{-12}$.

**Reciprocal-space sum** (smooth part, converges exponentially):

$$E_\mathrm{recip} = \frac{e^2}{2} \frac{4\pi}{\Omega} \sum_{\mathbf{G}\ne 0} \frac{e^{-|\mathbf{G}|^2/4\eta^2}}{|\mathbf{G}|^2} \, |S(\mathbf{G})|^2$$

where $S(\mathbf{G}) = \sum_i Z_i e^{-i\mathbf{G}\cdot\mathbf{r}_i}$ is the ionic structure factor.  The reciprocal cutoff is $G_\mathrm{cut}^2 = -4\eta^2 \ln\epsilon$.

**Self-energy correction** (removes spurious self-interaction):

$$E_\mathrm{self} = -e^2 \frac{\eta}{\sqrt{\pi}} \sum_i Z_i^2$$

**Charged-cell correction** (nonzero only for non-neutral cells):

$$E_\mathrm{charged} = -\frac{e^2}{2} \frac{\pi}{\Omega\eta^2} \left(\sum_i Z_i\right)^2$$

---

## 7. Hellmann-Feynman Forces

At SCF convergence, the force on atom $I$ is:

$$\mathbf{F}_I = -\frac{\partial E_\mathrm{tot}}{\partial \mathbf{R}_I}$$

The total force decomposes into three contributions: Ewald, local PP, and nonlocal PP.

### 7.1 Ewald forces

**Real-space contribution.** Differentiating $E_\mathrm{real}$ with respect to $\mathbf{r}_i$:

$$\mathbf{F}_i^\mathrm{real} = -e^2 \sideset{}{'}\sum_{j,\mathbf{T}} Z_i Z_j \left[\frac{\operatorname{erfc}(\eta r)}{r^2} + \frac{2\eta}{\sqrt\pi} \frac{e^{-\eta^2 r^2}}{r}\right] \frac{\mathbf{r}_{ij}+\mathbf{T}}{r}$$

where $r = |\mathbf{r}_j - \mathbf{r}_i + \mathbf{T}|$.  The factor of 2 from the double sum $(i,j)$ and $(j,i)$ cancels the $1/2$ prefactor.

**Reciprocal-space contribution.** The derivative acts on $|S(\mathbf{G})|^2$ through the phase factor of atom $i$:

$$\mathbf{F}_i^\mathrm{recip} = \frac{e^2 \cdot 4\pi}{\Omega} Z_i \sum_{\mathbf{G}\ne 0} \frac{e^{-G^2/4\eta^2}}{G^2} \, \mathbf{G} \left[S_r \sin(\mathbf{G}\cdot\mathbf{r}_i) + S_i \cos(\mathbf{G}\cdot\mathbf{r}_i)\right]$$

where $S(\mathbf{G}) = S_r + i S_i$.

### 7.2 Local PP forces

$$\mathbf{F}_I^\mathrm{loc} = -\frac{\partial}{\partial\mathbf{R}_I} \sum_\mathbf{G} V_\mathrm{loc}(\mathbf{G}) \, n^*(\mathbf{G})$$

The position dependence enters through the structure factor $S(\mathbf{G})$ in $V_\mathrm{loc}(\mathbf{G})$.  The derivative of $e^{-i\mathbf{G}\cdot\boldsymbol{\tau}_I}$ produces a factor of $-i\mathbf{G}$, giving:

$$F_I^{\mathrm{loc}}[d] = \Omega \sum_{\mathbf{G}\ne 0} V_\mathrm{loc}^s(|\mathbf{G}|) \, G_d \left[n_r \sin(\mathbf{G}\cdot\boldsymbol{\tau}_I) + n_i \cos(\mathbf{G}\cdot\boldsymbol{\tau}_I)\right]$$

where $n(\mathbf{G}) = n_r + i n_i$ and $V_\mathrm{loc}^s$ is the per-species form factor (excluding the structure factor).  The $\mathbf{G} = 0$ term vanishes identically.

### 7.3 Nonlocal PP forces

$$\mathbf{F}_I^\mathrm{NL} = -\sum_{n\mathbf{k}} f_{n\mathbf{k}} \frac{\partial}{\partial\mathbf{R}_I} \langle\psi_{n\mathbf{k}}|\hat{V}_\mathrm{NL}|\psi_{n\mathbf{k}}\rangle$$

The derivative acts on the projectors $\beta_i^a$ through the phase $e^{-i(\mathbf{k}+\mathbf{G})\cdot\boldsymbol{\tau}_a}$:

$$\frac{\partial \beta_i^a(\mathbf{G})}{\partial \tau_a^d} = -i(k+G)_d \, \beta_i^a(\mathbf{G})$$

Defining $P_j = \langle\beta_j|\psi\rangle$ and $\partial P_i / \partial\tau^d = i \sum_\mathbf{G} (k+G)_d \, \beta_i^*(\mathbf{G}) \, \psi(\mathbf{G})$, the force becomes:

$$F_I^{\mathrm{NL}}[d] = -\sum_{n\mathbf{k}} f_{n\mathbf{k}} \sum_{i,j} D_{ij} \cdot 2\,\mathrm{Re}\!\left[\overline{P_j} \frac{\partial P_i}{\partial\tau^d}\right]$$

### 7.4 Pulay forces

For a complete (converged) plane-wave basis, Pulay forces vanish exactly because the basis does not depend on atomic positions.  This is a major advantage over localized basis sets.

### 7.5 Force symmetrization

When spglib is available, KRONOS symmetrizes the computed forces under the full crystal point group.  For each symmetry operation $(R, \mathbf{t})$:

$$\mathbf{F}_i^\mathrm{sym} = \frac{1}{N_\mathrm{ops}} \sum_{\mathrm{ops}} R_\mathrm{cart} \, \mathbf{F}_j^\mathrm{raw}$$

where atom $j$ maps to atom $i$ under the operation.  This enforces the symmetry that is broken by using IBZ k-points with different active plane-wave counts.

---

## 8. SCF Convergence Methods

### 8.1 Davidson iterative eigensolver

KRONOS uses the block Davidson iterative diagonalization to find the lowest $N_\mathrm{bands}$ eigenvalues of $H|\psi\rangle = \varepsilon|\psi\rangle$ without forming or storing $H$ explicitly.

Algorithm outline for each k-point:

1. Start with $N_\mathrm{bands}$ trial vectors $\{|v_i\rangle\}$
2. Apply $H$ to all trial vectors: $|Hv_i\rangle$
3. Build the projected (Rayleigh-Ritz) matrix $H_{ij}^\mathrm{sub} = \langle v_i|H|v_j\rangle$ and diagonalize
4. Compute residuals $|r_i\rangle = (H - \theta_i)|u_i\rangle$ where $\theta_i, |u_i\rangle$ are Ritz values/vectors
5. Precondition: $|\delta_i\rangle = (T_\mathbf{G} - \theta_i)^{-1}|r_i\rangle$ using the kinetic-energy diagonal
6. Expand the subspace: add $\{|\delta_i\rangle\}$ to the trial set (Gram-Schmidt orthogonalize)
7. Repeat from step 2 until $\|r_i\| < \mathrm{tol}$ for all wanted states

The subspace dimension is capped at $3 \times N_\mathrm{bands}$; when exceeded, the basis is collapsed back to the current Ritz vectors.  If the Davidson solver diverges (residual $> 10^3$), KRONOS auto-switches to LOBPCG for that k-point.

### 8.2 DIIS / Pulay density mixing

The SCF loop mixes input and output densities to damp charge sloshing.  KRONOS uses Pulay (DIIS) mixing with a history of $M = 8$ steps and a linear mixing parameter $\alpha = 0.2$.

Given input densities $\{n_i^\mathrm{in}\}$ and residuals $R_i = n_i^\mathrm{out} - n_i^\mathrm{in}$, DIIS finds coefficients $\{c_i\}$ that minimize $\|\sum_i c_i R_i\|^2$ subject to $\sum_i c_i = 1$.  This involves solving the linear system:

$$\begin{pmatrix} B_{11} & \cdots & B_{1M} & 1 \\ \vdots & \ddots & \vdots & \vdots \\ B_{M1} & \cdots & B_{MM} & 1 \\ 1 & \cdots & 1 & 0 \end{pmatrix} \begin{pmatrix} c_1 \\ \vdots \\ c_M \\ \lambda \end{pmatrix} = \begin{pmatrix} 0 \\ \vdots \\ 0 \\ 1 \end{pmatrix}$$

where $B_{ij} = \langle R_i | R_j \rangle$.  The new input density is:

$$n^\mathrm{in}_\mathrm{new} = \sum_i c_i \bigl[n_i^\mathrm{in} + \alpha \, R_i\bigr]$$

### 8.3 Kerker preconditioning

For metals, a **Kerker preconditioner** suppresses long-wavelength charge sloshing by filtering the residual in G-space:

$$R_\mathrm{precond}(\mathbf{G}) = R(\mathbf{G}) \cdot \frac{|\mathbf{G}|^2}{|\mathbf{G}|^2 + q_0^2}$$

This damps the $\mathbf{G} \to 0$ components that cause the metallic screening instability.  KRONOS uses $q_0 = 1.5$ bohr$^{-1}$ and activates Kerker preconditioning automatically when Gaussian or Fermi-Dirac smearing is enabled.

### 8.4 Convergence criteria

KRONOS checks both criteria each SCF step:

- **Energy convergence**: $|E^{(i)} - E^{(i-1)}| < \varepsilon_E$ (primary criterion, default $10^{-6}$ Ry)
- **Density convergence**: $\|\Delta n\|_\mathrm{G} / N_e < \varepsilon_n$ (secondary criterion, G-space L2 norm)

Energy convergence is the more physically meaningful measure: it is variational and directly determines the accuracy of total energies and forces.  The density residual is computed in G-space (PW components only) to avoid aliasing artifacts from the real-space grid.

The SCF loop aborts with a diagnostic if energy oscillates by more than 1 Ry between consecutive steps after step 15 (indicating a fundamental problem such as incorrect pseudopotential or basis).

### 8.5 Initial density

The initial electron density is constructed from the superposition of atomic charge densities read from UPF files:

$$n_0(\mathbf{G}) = \frac{1}{\Omega} \sum_s \rho_s^\mathrm{atom}(|\mathbf{G}|) \, S_s(\mathbf{G})$$

where $\rho_s^\mathrm{atom}(q) = \int \rho_s^\mathrm{UPF}(r) \, \mathrm{sinc}(qr) \, dr$ is the radial Fourier transform of the UPF atomic density (which stores $4\pi r^2 \rho(r)$).  If no atomic densities are available, a uniform density $n_0 = N_e / \Omega$ is used.

---

## References

1. P. Hohenberg and W. Kohn, Phys. Rev. 136, B864 (1964).
2. W. Kohn and L. J. Sham, Phys. Rev. 140, A1133 (1965).
3. J. P. Perdew and A. Zunger, Phys. Rev. B 23, 5048 (1981).
4. L. Kleinman and D. M. Bylander, Phys. Rev. Lett. 48, 1425 (1982).
5. P. P. Ewald, Ann. Phys. 369, 253 (1921).
6. H. J. Monkhorst and J. D. Pack, Phys. Rev. B 13, 5188 (1976).
7. P. Pulay, Chem. Phys. Lett. 73, 393 (1980).
8. E. R. Davidson, J. Comput. Phys. 17, 87 (1975).
9. G. P. Kerker, Phys. Rev. B 23, 3082 (1981).
10. J. P. Perdew, K. Burke, and M. Ernzerhof, Phys. Rev. Lett. 77, 3865 (1996).
