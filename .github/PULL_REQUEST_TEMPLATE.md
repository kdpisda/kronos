<!--
Thanks for the contribution! A short, focused PR description and the checklist below
help maintainers review faster. None of the boxes are blocking — just check the ones
that apply. Items that don't apply, strike through or remove.
-->

## What this PR does

<!-- 1–3 sentences. What changes, and why. -->

## Why now

<!-- What motivated this? Link to the issue or Discussions thread if there is one. -->
Closes #

## Type of change

<!-- Tick all that apply -->

- [ ] Bug fix (non-breaking)
- [ ] New feature (non-breaking)
- [ ] Performance / numerical accuracy improvement
- [ ] Refactor (no behavior change)
- [ ] Documentation only
- [ ] Build / CI / tooling
- [ ] Breaking change (please justify in the description)

## How it was tested

<!--
Be specific. "ran ctest" is fine if you also describe what scenarios you exercised.
- [ ] `ctest -j2` passes locally (CPU build)
- [ ] Tested on GPU backend: cuda / hip / metal (delete those not applicable)
- [ ] New regression test added for this behavior (file/test name)
- [ ] Validated against reference numbers — describe (e.g., QE comparison within X meV/atom)
-->

## Checklist

- [ ] Code follows the project's C++ conventions (see `docs/developer_guide.md`)
- [ ] New numerical code preserves `complex_t = std::complex<double>` (no silent fp32 narrowing on the CPU/CUDA/HIP paths)
- [ ] New code paths have at least one regression test in `test/`
- [ ] Documentation updated where user-visible behavior changes (`docs/`, `website/src/data/benchmarks.json`, READMEs)
- [ ] `CHANGELOG.md` updated under "Unreleased" (when present)
- [ ] No `--no-verify` or hook-bypass commits in this branch's history

## Numerical / physics impact

<!--
If this PR changes any computed number (total energy, forces, eigenvalues, band gap,
benchmark output, etc.), document the magnitude of the change and which reference
calculation you compared against. "No numerical impact" is also a valid answer.
-->

## Anything reviewers should pay extra attention to

<!--
Risky areas, places you're uncertain about, follow-ups intentionally left for later, etc.
-->
