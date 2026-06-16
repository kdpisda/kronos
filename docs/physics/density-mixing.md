---
title: Density Mixing — Pulay, DIIS, and Kerker
description: Charge sloshing in SCF, Pulay's DIIS residual minimization, the Kerker preconditioner for metals, and convergence diagnostics in KRONOS.
keywords:
  - density mixing DFT
  - Pulay mixing
  - DIIS density mixing
  - Kerker preconditioner
  - SCF convergence
  - charge sloshing
  - plane-wave DFT mixing
  - DFT convergence
  - residual minimization
  - linear mixing
slug: /physics/density-mixing
sidebar_position: 9
---

# Density Mixing — Pulay, DIIS, and Kerker

The Kohn-Sham equations are nonlinear: the effective potential depends on the density, which depends on the orbitals, which depend on the potential. Solving this fixed point naively almost never converges. Density mixing is the numerical machinery that turns a divergent fixed-point iteration into a robust SCF loop, and the choice of mixer is often the difference between convergence in 15 steps and convergence never.

## The fixed-point problem

A single self-consistent-field iteration maps an input density to an output density:

$$n_\mathrm{in} \xrightarrow{\text{KS solve}} n_\mathrm{out} = \mathcal{F}[n_\mathrm{in}]$$

The converged density is the fixed point $n^* = \mathcal{F}[n^*]$. The naive iteration $n^{(k+1)} = n_\mathrm{out}^{(k)}$ almost always oscillates or diverges because of **charge sloshing**: small changes in the density at long wavelengths (small $|\mathbf{G}|$) produce large changes in the Hartree potential ($V_\mathrm{H}(\mathbf{G}) \propto n(\mathbf{G})/G^2$), which feed back into a large output-density swing the other way.

Mixing combines $n_\mathrm{in}^{(k)}$ with $n_\mathrm{out}^{(k)}$ (and possibly earlier iterates) to damp this feedback loop into something that converges.

## Linear mixing

The simplest mixer is a weighted average:

$$n^{(k+1)} = (1-\alpha)\, n_\mathrm{in}^{(k)} + \alpha\, n_\mathrm{out}^{(k)}, \qquad \alpha \in (0, 1)$$

A small $\alpha$ (typically $0.2$–$0.3$) damps oscillation but yields slow convergence — often hundreds of SCF iterations for metals. Linear mixing is robust but not competitive; it serves as a fallback and as a baseline against which fancier mixers are measured.

## Residual minimization

Define the SCF residual:

$$R^{(k)} = n_\mathrm{out}^{(k)} - n_\mathrm{in}^{(k)}$$

At the fixed point $R = 0$. Pulay's insight (1980) was that, given a *history* of input densities and their residuals, we can construct a linear combination

$$n_\mathrm{in}^* = \sum_{i=1}^{M} c_i\, n_\mathrm{in}^{(i)}, \qquad \sum_i c_i = 1$$

whose **predicted** residual

$$R^* = \sum_i c_i\, R^{(i)}$$

has minimum norm. The optimal coefficients $\{c_i\}$ extrapolate toward $R = 0$ using only iterates we have already computed — no extra Hamiltonian builds, just linear algebra on a small $(M+1)\times(M+1)$ matrix.

## Pulay's DIIS equations

Minimizing $\| R^* \|^2$ subject to $\sum c_i = 1$ with a Lagrange multiplier $\lambda$ yields the linear system

$$\begin{pmatrix} B_{11} & \cdots & B_{1M} & 1 \\ \vdots & & \vdots & \vdots \\ B_{M1} & \cdots & B_{MM} & 1 \\ 1 & \cdots & 1 & 0 \end{pmatrix} \begin{pmatrix} c_1 \\ \vdots \\ c_M \\ \lambda \end{pmatrix} = \begin{pmatrix} 0 \\ \vdots \\ 0 \\ 1 \end{pmatrix}$$

with the overlap matrix $B_{ij} = \langle R^{(i)} | R^{(j)} \rangle$ (the inner product taken in real space or, in KRONOS, in G-space for numerical robustness). Once $\{c_i\}$ are obtained, the mixed input density for the next SCF iteration is

$$n^{(k+1)} = \sum_{i=1}^{M} c_i \left[ n_\mathrm{in}^{(i)} + \alpha\, R^{(i)} \right]$$

The factor $\alpha$ here is a small linear-mixing step applied *inside* the DIIS extrapolation — it stabilizes the procedure when the residual history is short and the matrix $B$ is ill-conditioned.

## History depth and stability

KRONOS uses a default history depth of $M = 8$. Larger $M$ in principle improves the rate of convergence, but $B$ tends to become ill-conditioned because successive residuals get nearly parallel as the SCF converges. KRONOS detects this and falls back to using only the most recent entry when the smallest pivot in the LU factorization drops below $10^{-15}$:

```
if (min_pivot < 1e-15) {
    // matrix singular — drop oldest, retry with shorter history
}
```

Without this safeguard, a near-converged SCF can suddenly diverge as numerical noise in $B$ produces extrapolation coefficients of order $\pm 10^3$.

## Kerker preconditioning for metals

Charge sloshing in metals is dominated by long wavelengths — the divergent $1/G^2$ in the Hartree response means small-$G$ residuals are over-weighted in $B_{ij}$. Kerker's fix (1981) is to precondition the residual in G-space before mixing:

$$R(\mathbf{G}) \to R(\mathbf{G}) \cdot \frac{G^2}{G^2 + q_0^2}$$

The screening wavevector $q_0$ is typically chosen near the Thomas-Fermi screening length. KRONOS uses $q_0 = 1.5\,\mathrm{bohr}^{-1}$, which is appropriate for most metals. The preconditioner suppresses the small-$G$ components of the residual, so DIIS no longer "sees" the charge-sloshing modes and converges in 15–30 steps even for transition metals.

For insulators, the Kerker preconditioner is unnecessary (no Fermi surface, no sloshing) and may slow convergence. KRONOS activates Kerker only when `smearing` is non-`None`, i.e., when the calculation is treating a metal.

## Convergence criteria

KRONOS declares convergence when **both** of these are satisfied for a single SCF step:

1. **Energy:** $|E^{(k)} - E^{(k-1)}| < \epsilon_E$
2. **Density:** $\| n_\mathrm{out}^{(k)} - n_\mathrm{in}^{(k)} \|_G < \epsilon_n$

The density norm is computed in G-space to avoid aliasing artifacts from the real-space grid. Defaults are $\epsilon_E = 10^{-7}\,\mathrm{Ry}$ and $\epsilon_n = 10^{-6}$, controllable via `convergence.energy_threshold` and `convergence.density_threshold` in the YAML input.

A non-converged run after `max_scf_steps` (default 200) writes a partial output marked `"converged": false` and emits a diagnostic suggesting tighter `ecutwfc`, denser k-grid, or smaller mixing $\alpha$.

## Initial density and convergence speed

The closer the initial guess to the fixed point, the fewer SCF iterations are required. KRONOS initializes the density as a superposition of atomic densities, $n^{(0)}(\mathbf{r}) = \sum_a \rho_a^\mathrm{atom}(|\mathbf{r} - \boldsymbol\tau_a|)$, using the `rho_atomic` data field from the UPF file. When a pseudopotential lacks `rho_atomic` (some older HGH files), the initial density falls back to uniform over the cell — typically adding 5–10 SCF iterations.

## How KRONOS implements this

The mixer lives in `src/solver/mixing.cpp` as the `PulayMixer` class. Its responsibilities, in order:

1. Push each iteration's $(n_\mathrm{in}^{(k)}, R^{(k)})$ pair into a ring buffer of depth $M$.
2. Apply the Kerker preconditioner to each residual when metallic smearing is active.
3. Build the $B$ matrix in G-space (Parseval): $B_{ij} = \Omega \sum_\mathbf{G} R_i(\mathbf{G})^* R_j(\mathbf{G})$.
4. Solve the augmented linear system, falling back to shorter history if the pivot threshold is crossed.
5. Construct the mixed input density, clamp $n(\mathbf{r}) \geq 0$, and renormalize to conserve total electron count.

After the mix, the loop returns to step (a) of the SCF flow at [SCF Flowchart](/docs/architecture/scf-flowchart). The data structures and call sites are documented in the architecture page on [Algorithms](/docs/architecture/algorithms).

## References

- Pulay, P. "Convergence acceleration of iterative sequences. The case of SCF iteration", *Chem. Phys. Lett.* **73**, 393 (1980).
- Kerker, G. P. "Efficient iteration scheme for self-consistent pseudopotential calculations", *Phys. Rev. B* **23**, 3082 (1981).
- Kresse, G. & Furthmüller, J. "Efficiency of ab-initio total energy calculations for metals and semiconductors", *Comput. Mater. Sci.* **6**, 15 (1996).
- Johnson, D. D. "Modified Broyden's method for accelerating convergence in self-consistent calculations", *Phys. Rev. B* **38**, 12807 (1988).
- Martin, R. M. *Electronic Structure: Basic Theory and Practical Methods*, Cambridge University Press, 2004 — Ch. 9.
