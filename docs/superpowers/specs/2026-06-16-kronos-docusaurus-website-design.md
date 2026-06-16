# KRONOS Documentation Website — Design Spec

**Date:** 2026-06-16
**Author:** Kuldeep Pisda (via Claude brainstorming session)
**Status:** Draft, pending implementation plan
**Target version:** KRONOS website v1 (initial public site)

## 1. Motivation

KRONOS is a research-grade DFT engine. Its credibility against established
codes (Quantum ESPRESSO, Wien2k, VASP) is the single biggest factor for
adoption by computational physics researchers. The repo's existing 2,700+
lines of markdown documentation are not discoverable and don't surface the
QE-agreement story that makes KRONOS trustworthy.

A real documentation website with a benchmarks page that leads with QE
comparison numbers turns "trust me, the README says it works" into "here are
the numbers — KRONOS agrees with QE to 0.07 meV/atom on Si LDA." That's the
shift from "interesting project" to "credible engine."

Goal: a Docusaurus site at `kdpisda.github.io/kronos` whose landing page and
benchmarks page exist primarily to build user confidence, with the rest of
the existing documentation organized into a navigable site.

## 2. Constraints

### 2.1 Existing content must not duplicate
The repo already has ~2,700 lines of well-written markdown in `docs/*.md`
(architecture, user_guide, developer_guide, physics_notes, api_reference)
plus README.md and VALIDATION.md. These are the canonical sources. The
website MUST read from them directly — no copies, no parallel-source
maintenance burden.

### 2.2 Minimal infrastructure dependency
The project owner is a solo maintainer. The website setup should not
introduce a SaaS dependency (no Algolia/Vercel signup required for v1)
beyond GitHub itself.

### 2.3 Benchmark numbers must be real and verifiable
The benchmarks page is the trust pitch. Numbers shown there must come from
actual validated runs (currently captured in `test/test_validation.cpp` and
the project memory notes), not placeholder estimates. Source citations
matter: each number should link to or reference the test that produced it.

### 2.4 No new content writing in this scope
The website project organizes and presents existing content. Writing new
tutorials, fresh benchmark runs against QE, or expanded API docs is OUT of
scope and tracked separately.

## 3. Site Information Architecture

### 3.1 Sidebar layout

| Section | Source |
|---|---|
| **Getting Started** — intro, install, first run | new (extracted from `README.md`) |
| **User Guide** | existing `docs/user_guide.md` (split into sub-pages where natural) |
| **Tutorials** — Si bulk, Cu metal, H₂O, Fe BCC | placeholder pages now (point to `examples/*.yaml`) |
| **Developer Guide** | existing `docs/developer_guide.md` + `docs/architecture.md` |
| **Physics Notes** | existing `docs/physics_notes.md` |
| **Benchmarks** | new page, see §5 |
| **API Reference** | existing `docs/api_reference.md` |
| **How to Cite** | new (reads `CITATION.cff` + BibTeX block) |
| **Contributing** | new (mirrors `CONTRIBUTING.md` from repo) |
| **Community** | new (Discussions, Code of Conduct, Security policy) |
| **Roadmap** | new (semver milestones, current targets) |
| **Publications** | new (placeholder: "papers using KRONOS — PRs welcome") |
| **Changelog** | new (mirrors `CHANGELOG.md` from repo, generated from git tags) |

### 3.2 Landing page (`website/src/pages/index.tsx`)

Hero band:
- Single one-line tagline: *"Plane-wave DFT, validated against Quantum ESPRESSO to meV precision."*
- Three credibility numbers, large type:
  - **0.07 meV/atom** — Si LDA Γ-only vs QE
  - **0.15 meV/atom** — Si 4×4×4 shifted vs QE
  - **5 sig figs** — forces match by finite-difference
- Two CTA buttons: "View benchmarks" → `/benchmarks`, "Get started" → `/getting-started`

Below hero:
- 3-tile feature grid:
  - **Validated** — < 2 meV/atom Delta-test target on the systems we test
  - **GPU-accelerated** — CUDA, HIP, Metal backends (Apple Silicon supported in fp32)
  - **MPI-parallel** — k-point and band parallelism, scaling to research clusters
- A 4-line code snippet showing a minimal Si bulk YAML input.

Footer:
- GPL-3 license link
- GitHub repo link
- **How to cite** link → `/cite`
- **Contributing** link → `/contributing`
- **Community / Discussions** link → GitHub Discussions
- Author line

Below the footer (above the fold's secondary band), a single line of badges:
CI status · License: GPL-3 · Latest release · DOI (when minted) · Stars.

No animations. No marketing copy beyond the hero band.

## 4. Architecture

### 4.1 Project layout

```
kronos/                                # repo root
├── website/                           # Docusaurus project (NEW)
│   ├── docusaurus.config.ts
│   ├── sidebars.ts
│   ├── package.json
│   ├── tsconfig.json
│   ├── src/
│   │   ├── pages/
│   │   │   └── index.tsx              # custom landing page
│   │   ├── components/
│   │   │   ├── BenchmarkHero.tsx      # 3-number credibility band
│   │   │   ├── BenchmarkTable.tsx     # systems table
│   │   │   ├── BenchmarkChart.tsx     # Recharts bar chart
│   │   │   └── FeatureCard.tsx        # landing tile
│   │   ├── data/
│   │   │   └── benchmarks.json        # single source of truth for numbers
│   │   └── css/
│   │       └── custom.css
│   └── static/
│       └── img/                       # logo wordmark, og-image
├── docs/                              # existing — Docusaurus reads from here
│   ├── architecture.md                # → Developer Guide
│   ├── user_guide.md                  # → User Guide
│   ├── developer_guide.md             # → Developer Guide
│   ├── physics_notes.md               # → Physics Notes
│   └── api_reference.md               # → API Reference
├── benchmarks/                        # NEW — reproducibility scaffolding
│   ├── README.md                      # how to run + add benchmarks
│   ├── si-bulk-lda-gamma/
│   │   ├── kronos.yaml
│   │   ├── pw.in                      # QE input
│   │   ├── run.sh                     # runs both, captures timings
│   │   └── README.md
│   ├── si-bulk-lda-444-shifted/
│   ├── cu-fcc-pbe/
│   └── (more systems as data lands)
└── .github/
    └── workflows/
        └── docs.yml                   # NEW — build + deploy to gh-pages
```

Docusaurus is configured so `presets.classic.docs.path = '../docs'` — the
existing markdown is treated as the content source. The `website/`
directory contains only the framework + custom code, never the canonical
content.

### 4.2 Tech stack

| Dependency | Purpose | Notes |
|---|---|---|
| Docusaurus 3.x (latest stable) | Static site generator | TypeScript config |
| React 18 | Component model | Default via Docusaurus |
| MDX | Inline React in markdown | For embedding the benchmark chart |
| `@docusaurus/theme-mermaid` | Architecture diagrams | Built-in plugin |
| `remark-math` + `rehype-katex` | LaTeX in physics notes | Standard Docusaurus math |
| `recharts` | Benchmark bar chart | ~30 KB gzipped, no D3 |
| `@easyops-cn/docusaurus-search-local` | Local search | No Algolia dependency |
| Node.js 20+ | Build | Pinned in `package.json` engines |

### 4.3 Build & deploy

GitHub Actions workflow `.github/workflows/docs.yml`:
1. Trigger: push to `main` that changes `docs/**`, `website/**`, `benchmarks/**`, or `README.md`
2. Steps: checkout → Node 20 setup → `npm ci` → `npm run build` → deploy to `gh-pages` branch via `peaceiris/actions-gh-pages` (the standard action).
3. URL: `https://kdpisda.github.io/kronos`
4. CNAME: not configured in this scope (custom domain is a later decision).

## 5. Benchmarks Page Design

The page is the centerpiece of the credibility pitch.

### 5.1 Page structure

```
┌─────────────────────────────────────────────────┐
│ Hero: 3 large headline numbers                  │
│ 0.07 meV/atom · 0.15 meV/atom · 5 sig figs      │
├─────────────────────────────────────────────────┤
│ Methodology paragraph (4-5 sentences)           │
│ - What "Delta test" means                       │
│ - How errors are measured                       │
│ - Toy PPs vs real ONCV note                     │
├─────────────────────────────────────────────────┤
│ System-by-system table:                         │
│ ┌─────┬────────┬─────────┬────────┬─────────┐  │
│ │ Sys │ Method │ KRONOS  │ QE     │ Δ meV/at│  │
│ ├─────┼────────┼─────────┼────────┼─────────┤  │
│ │ Si  │ LDA Γ  │ -28.6052│ ...    │ 0.07    │  │
│ │ Si  │ 2×2×2  │ -28.0558│ ...    │ ...     │  │
│ │ ... │        │         │        │         │  │
│ └─────┴────────┴─────────┴────────┴─────────┘  │
├─────────────────────────────────────────────────┤
│ Bar chart: per-system error (meV/atom)         │
│ Recharts horizontal bars                        │
├─────────────────────────────────────────────────┤
│ Reproducibility section                         │
│ - Link to benchmarks/ directory                 │
│ - How to run yourself                           │
│ - "Fresh data PRs welcome"                      │
└─────────────────────────────────────────────────┘
```

### 5.2 Data source

All numbers live in `website/src/data/benchmarks.json`. Schema:

```jsonc
{
  "headline": [
    {"label": "Si LDA Γ-only", "value": 0.07, "unit": "meV/atom", "source": "test_validation.cpp::QEComparison.SiLDAGammaToyPP"},
    {"label": "Si 4×4×4 shifted", "value": 0.15, "unit": "meV/atom", "source": "test_validation.cpp::QEComparison.Si444ShiftedToyPP"},
    {"label": "Forces vs QE", "value": "5 sig figs", "unit": null, "source": "VALIDATION.md"}
  ],
  "systems": [
    {
      "system": "Si bulk diamond",
      "method": "LDA Γ-only",
      "kronos_total_energy_ry": -28.6052,
      "qe_total_energy_ry": -28.6053,
      "delta_mev_per_atom": 0.07,
      "source": "test_validation.cpp::SiGammaLDA"
    },
    // ... rows for Si 2×2×2, Si 4×4×4, Al FCC, Cu FCC, H₂O, MgO, graphene, Fe BCC LSDA
  ]
}
```

The React components `BenchmarkHero`, `BenchmarkTable`, `BenchmarkChart` all
consume this single JSON file. Updating numbers = editing one file.

Numbers come from existing test results captured in `test_validation.cpp`
and the project memory; the spec lists them but the implementation plan
will extract the exact current values from the test source.

### 5.3 Reproducibility scaffolding

New `benchmarks/` directory at the repo root. Each system gets a
subdirectory with:
- `kronos.yaml` — the input file
- `pw.in` — the matching QE input
- `run.sh` — runs both, extracts total energies, prints comparison
- `README.md` — what the system is, expected agreement, citation if any

Systems in v1 (matching the table above) have the subdirectories created
with proper `kronos.yaml` (matching what `test_validation.cpp` constructs)
but the `pw.in` and `run.sh` are stubs marked TODO. The website page links
to the GitHub directory and says "PRs to fill these in are welcome."

This is the explicit handle for future "real fresh QE runs" work.

## 6. Open-Source Hygiene & Community

A serious open-source research project needs more than docs. The website is
one part — the repo-level files and GitHub features are the other. This
section enumerates both.

### 6.1 Repo-level files (NEW)

| File | Purpose | Notes |
|---|---|---|
| `CONTRIBUTING.md` | How to contribute: branch model, build/test, PR expectations, commit message convention | One page. Sources: existing `docs/developer_guide.md` + project conventions |
| `CODE_OF_CONDUCT.md` | Contributor Covenant 2.1 | Standard text, drop-in |
| `SECURITY.md` | How to report security issues (vulnerable inputs, malformed UPF files, etc.) | Email contact, GitHub security advisories preference |
| `CITATION.cff` | Machine-readable citation metadata (GitHub renders a "Cite this repository" button) | Maps to author, version, DOI (when minted) |
| `CHANGELOG.md` | Keep-a-Changelog format | Seeded from current git tags (v0.5.1-rc1) |
| `.github/ISSUE_TEMPLATE/bug_report.yml` | Structured bug reports | Form template, not free-text |
| `.github/ISSUE_TEMPLATE/feature_request.yml` | Structured feature asks | |
| `.github/ISSUE_TEMPLATE/question.yml` | Steer general questions to Discussions | Points users away from issues |
| `.github/ISSUE_TEMPLATE/config.yml` | Disables blank issues, links to Discussions | |
| `.github/PULL_REQUEST_TEMPLATE.md` | PR checklist (tests, docs, CHANGELOG) | |
| `.github/FUNDING.yml` | GitHub Sponsors / Ko-fi (if author wants donations) | Optional — leave empty if not |

### 6.2 GitHub features to enable

- **Discussions** — turn on, set categories (Q&A, Show & Tell, Ideas, Polls, Announcements). Become the community forum so issues stay actionable.
- **Releases** — formal releases tied to semver tags. `v0.5.1-rc1` already exists; v1 releases should attach a compiled binary set or at least a tarball.
- **Branch protection on `main`** — require PR review, CI green, no force-push. Documents the rules everyone follows.
- **Security policy** — link `SECURITY.md` via repo Security tab.
- **Code owners** (`.github/CODEOWNERS`) — at least for the GPU and SCF directories so any change pings the right reviewer. Currently `@kdpisda` owns everything.

### 6.3 README badges

Top of `README.md` after the title, single line:

```
[![CI](https://github.com/kdpisda/kronos/actions/workflows/ci.yml/badge.svg)](...)
[![Docs](https://github.com/kdpisda/kronos/actions/workflows/docs.yml/badge.svg)](...)
[![License](https://img.shields.io/badge/license-GPL--3-blue.svg)](LICENSE)
[![Release](https://img.shields.io/github/v/release/kdpisda/kronos)](releases)
[![DOI](https://zenodo.org/badge/DOI/...)](...)        <!-- added when Zenodo mints one -->
```

Real numbers (CI green, release tag) — not vanity placeholders.

### 6.4 Website pages tied to OSS hygiene

These exist on the site to make the repo-level files discoverable:

- **`/cite`** — renders BibTeX block + a "Cite this repo" button that copies the BibTeX. Reads metadata from `CITATION.cff` (parsed at build time).
- **`/contributing`** — full content of `CONTRIBUTING.md` (Docusaurus reads the file as a page).
- **`/community`** — links to GitHub Discussions, Code of Conduct, Security policy. Plain text plus three big tiles.
- **`/roadmap`** — versioning roadmap (v0.5 → v0.8 → v1.0 → v2.0) with what's done and what's next.
- **`/publications`** — empty list of papers using KRONOS, with a "Submit yours via PR" CTA at the bottom.

### 6.5 DOI / Zenodo (handled separately)

Zenodo's GitHub integration mints a DOI for each tagged release. This
requires the project owner to authorize Zenodo on the GitHub account
(one-time, takes 2 minutes). The DOI lands in `CITATION.cff` and the
README badge after the next release. This is documented in the spec but
the authorization step is OUT of scope for the implementation plan — the
owner does it manually.

## 7. Out of Scope (explicitly)

- **Fresh KRONOS-vs-QE runs.** This spec ships the existing validated
  numbers. Running QE side-by-side on this machine — and adding wall-clock
  timings — is a separate effort tracked by the `benchmarks/` directory
  scaffolding.
- **Tutorials with full output.** Tutorials are placeholder pages that
  point at `examples/*.yaml` from the repo. Building rich tutorials with
  computed results, plots, and discussion is later.
- **Versioned docs.** Docusaurus supports them but adds complexity. Start
  unversioned (always-latest). Add versioning when v1.0 ships.
- **Algolia DocSearch.** Local search is enough for an initial site. Add
  Algolia (free tier for OSS) later if usage grows.
- **Custom logo / illustration design.** A simple text wordmark and the
  Docusaurus default theme are sufficient. Brand design later.
- **Custom domain.** `kdpisda.github.io/kronos` is fine. CNAME setup later.
- **Tutorials for hybrid functionals and PAW workflows.** Existing docs
  cover the basics; specialized tutorials are later content work.

Additional items added by the OSS hygiene work (§6):

- **Manual Zenodo authorization** — must be done by the repo owner via
  Zenodo's GitHub integration (~2 minutes), not by the implementation plan.
- **Actual community moderation policies, governance docs, RFC process** —
  premature for a solo-maintained project. Add when contributor count grows.
- **CONTRIBUTING.md content beyond the basics** — branch model, build/test,
  PR checklist. No deep guide writing in this scope.

## 8. Acceptance Criteria

1. `cd website && npm install && npm run start` produces a dev server at
   `localhost:3000` with all sidebar sections populated from existing docs.
2. `cd website && npm run build` produces a clean static site in
   `website/build/` with no broken links or build errors.
3. Landing page hero shows the three headline numbers prominently.
4. Benchmarks page table shows KRONOS vs QE numbers for at least the 4
   primary systems (Si LDA Γ-only, Si 4×4×4, Al FCC, Cu FCC).
5. Benchmarks bar chart renders without errors and reads from the same
   JSON file as the table.
6. Mathematical notation in `physics_notes.md` renders via KaTeX (the
   existing markdown likely uses dollar-delimited math).
7. Mermaid diagrams (if any in `architecture.md`) render via the Mermaid
   plugin.
8. Local search returns results for "PAW", "Davidson", "hybrid".
9. GitHub Actions workflow builds and deploys to `gh-pages` on push to
   `main`. URL `kdpisda.github.io/kronos` serves the site.
10. `benchmarks/` directory exists with at least 4 system subdirectories
    each containing a valid `kronos.yaml`, a `README.md`, and TODO stubs
    for `pw.in` + `run.sh`.
11. `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`, `CITATION.cff`,
    `CHANGELOG.md` exist at the repo root with real content (not just
    skeletons).
12. `.github/ISSUE_TEMPLATE/{bug_report.yml,feature_request.yml,question.yml,config.yml}`
    + `.github/PULL_REQUEST_TEMPLATE.md` exist and produce structured
    issue/PR forms on GitHub.
13. README has a working badge line (CI, Docs, License, Release).
14. Website's `/cite` page renders BibTeX derived from `CITATION.cff`
    with a copy button.
15. Website's `/contributing`, `/community`, `/roadmap`, `/publications`
    pages are reachable from the sidebar and render meaningful content.

## 9. Risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Existing markdown has Docusaurus-incompatible Markdown extensions | Med | Low | Test each file with `npm run build`; fix any incompatible syntax. Most likely culprits: HTML inside markdown, raw `<` characters. |
| KaTeX rendering of physics notes — existing math may use non-standard delimiters | Med | Med | Inspect `physics_notes.md`; convert to standard `$...$` and `$$...$$` if needed (separate small task). |
| Mermaid syntax in `architecture.md` is currently rendered by GitHub's markdown engine; Docusaurus Mermaid uses the same syntax but stricter parser | Low | Low | Render and visually inspect each diagram once the site builds. |
| Benchmark numbers extracted from `test_validation.cpp` are out of date or missed by the implementer | Med | Med | Implementation plan tasks the implementer to read each test, extract the assertion target, and cite the test name in `benchmarks.json`. |
| GitHub Actions workflow misconfigured — site never publishes | Low | High | First-deploy verification step: visit the live URL after the workflow runs. |

## 10. Deliverables

1. `website/` directory with a working Docusaurus 3 setup
2. Custom landing page emphasizing QE-agreement numbers + footer with cite/contributing/community links
3. Benchmarks page with hero, table, chart, reproducibility section
4. `benchmarks.json` data file
5. `benchmarks/` directory with 4+ system subdirectories (scaffolding only)
6. `.github/workflows/docs.yml` deploy workflow
7. Updated `README.md` with the badge line, link to the live site, and brief intro
8. All existing `docs/*.md` files reachable through the website sidebar
9. **OSS hygiene files** — `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`, `CITATION.cff`, `CHANGELOG.md`
10. **GitHub templates** — issue templates (bug, feature, question, config), PR template
11. **Website OSS pages** — `/cite`, `/contributing`, `/community`, `/roadmap`, `/publications`

## 11. Open Questions for the Implementation Plan

1. **Sidebar nesting depth** — split `user_guide.md` into sub-pages (e.g.,
   "YAML input reference" as its own page, "Pseudopotentials" as another),
   or keep it as a single long page? The split improves navigability; the
   monolith preserves the existing flow.
2. **Recharts vs an even smaller alternative** — Recharts is ~30 KB; pure
   SVG hand-coded bars would be ~2 KB. For a single chart, do we want the
   dependency? Default to Recharts unless the bundle size matters.
3. **Where the README's existing content goes** — full landing copy, or
   split so the GitHub README has a short intro and the site has the long
   one? Defer to plan-writing.
