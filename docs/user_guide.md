# KRONOS User Guide

## Quick Start

### Dependencies

Required:
- C++20 compiler (GCC 11+, Clang 14+, Apple Clang 15+)
- CMake 3.20+
- FFTW3
- BLAS + LAPACK
- yaml-cpp

Optional:
- libxc 6.0+ (built-in LDA fallback if absent)
- HDF5 (for binary output)
- MPI (for parallel calculations, v0.2+)

### Build

```bash
# Configure (CPU-only, tests enabled)
cmake -B build -S .

# Build
cmake --build build -j$(nproc)

# Run tests
cd build && ctest --output-on-failure
```

### Run a Calculation

```bash
./build/kronos examples/si_bulk.yaml
```

---

## YAML Input Reference

KRONOS reads a single YAML input file with strict schema validation. Unknown keys cause hard errors.

### Complete Example

```yaml
crystal:
  lattice:                    # Lattice vectors in Angstrom (rows)
    - [0.0, 2.715, 2.715]    # a1
    - [2.715, 0.0, 2.715]    # a2
    - [2.715, 2.715, 0.0]    # a3
  atoms:
    - symbol: Si
      position: [0.00, 0.00, 0.00]   # Fractional coordinates
    - symbol: Si
      position: [0.25, 0.25, 0.25]

calculation:
  type: scf                   # scf | relax | bands | dos
  ecutwfc: 30.0               # Plane-wave cutoff (Ry), range: 10-500
  ecutrho: 0                  # Density cutoff (Ry), 0=auto (4×ecutwfc for NC)
  xc_functional: LDA_PZ      # LDA_PZ | LDA_PW | PBE | PBEsol
  kpoints:
    grid: [4, 4, 4]           # Monkhorst-Pack grid
    shift: [0, 0, 0]          # Grid shift (0 or 1)
  smearing: none              # none | gaussian | marzari-vanderbilt | fermi-dirac
  degauss: 0.01               # Smearing width (Ry)
  spin_polarized: false
  eigensolver: davidson       # davidson | lobpcg

convergence:
  energy_threshold: 1e-8      # Energy convergence (Ry)
  density_threshold: 1e-9     # Density convergence
  max_scf_steps: 100          # Max SCF iterations (hard limit: 200)
  force_threshold: 1e-3       # Force threshold for relax (Ry/bohr)

pseudopotentials:
  Si: pseudo/Si.UPF           # Path to UPF file per element

hardware:
  use_gpu: false
  gpu_backend: none            # none | cuda | hip
  mpi_tasks: 1
```

### Field Reference

| Field | Type | Default | Range/Values | Description |
|-------|------|---------|-------------|-------------|
| `crystal.lattice` | 3×3 float | required | — | Lattice vectors in Angstrom |
| `crystal.atoms[].symbol` | string | required | element symbol | Atomic species |
| `crystal.atoms[].position` | 3-float | required | [0,1) | Fractional coordinates |
| `calculation.type` | string | `scf` | scf, relax, bands, dos | Calculation type |
| `calculation.ecutwfc` | float | 30.0 | 10-500 | Wavefunction cutoff (Ry) |
| `calculation.ecutrho` | float | 0 (auto) | ≥4×ecutwfc | Density cutoff (Ry) |
| `calculation.xc_functional` | string | `LDA_PZ` | LDA_PZ, LDA_PW, PBE, PBEsol | XC functional |
| `calculation.kpoints.grid` | 3-int | [1,1,1] | ≥1 each | MP k-point grid |
| `calculation.kpoints.shift` | 3-int | [0,0,0] | 0 or 1 | Grid shift |
| `calculation.smearing` | string | `none` | none, gaussian, marzari-vanderbilt, fermi-dirac | Occupation smearing |
| `calculation.degauss` | float | 0.01 | >0 | Smearing width (Ry) |
| `convergence.energy_threshold` | float | 1e-8 | >0 | SCF energy tolerance (Ry) |
| `convergence.density_threshold` | float | 1e-9 | >0 | SCF density tolerance |
| `convergence.max_scf_steps` | int | 100 | 1-200 | Maximum SCF iterations |
| `convergence.force_threshold` | float | 1e-3 | >0 | Force tolerance for relax |

---

## Calculation Types

### SCF (Self-Consistent Field)

Standard ground-state calculation. Outputs total energy, eigenvalues, forces.

```yaml
calculation:
  type: scf
  ecutwfc: 30.0
  kpoints:
    grid: [4, 4, 4]
```

### Relax (Geometry Optimization)

Relaxes atomic positions to minimize forces using BFGS.

```yaml
calculation:
  type: relax
  ecutwfc: 30.0
  kpoints:
    grid: [4, 4, 4]
convergence:
  force_threshold: 1e-3    # Ry/bohr
```

### Bands (Band Structure)

Non-self-consistent calculation along a k-path using converged potential from a prior SCF.

```yaml
calculation:
  type: bands
  ecutwfc: 30.0
```

### DOS (Density of States)

Computes DOS from eigenvalues on a dense k-grid.

```yaml
calculation:
  type: dos
  ecutwfc: 30.0
  smearing: gaussian
  degauss: 0.02
  kpoints:
    grid: [8, 8, 8]
```

---

## Pseudopotentials

KRONOS reads norm-conserving pseudopotentials in UPF (Unified Pseudopotential Format) v2.

### Recommended Sources

| Library | URL | Recommendation |
|---------|-----|---------------|
| PseudoDojo | http://www.pseudo-dojo.org | NC stringent (preferred) |
| SSSP | https://www.materialscloud.org/discover/sssp | Efficiency or precision |
| GBRV | https://www.physics.rutgers.edu/gbrv | Ultrasoft (when supported) |

### Requirements

- Format: UPF v2 (XML-based)
- Type: Norm-conserving (v0.1). Ultrasoft/PAW in future versions.
- Norm conservation check is mandatory on load
- XC functional in PP should match the calculation's `xc_functional`

---

## Output

### JSON Summary

Written to stdout. Key fields:

```json
{
  "converged": true,
  "scf_steps": 23,
  "total_energy_ry": -15.847623,
  "total_energy_ev": -215.624,
  "fermi_energy_ev": 6.342,
  "kinetic_energy": 6.234,
  "hartree_energy": 1.876,
  "xc_energy": -4.523,
  "local_pp_energy": -12.445,
  "nonlocal_pp_energy": 0.0,
  "ewald_energy": -8.234,
  "forces": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
  "eigenvalues": [[-0.342, -0.156, 0.234, 0.567]],
  "timing": {"total": 12.4, "fft": 3.2, "davidson": 7.1}
}
```

### HDF5 Files

Binary output (if HDF5 is available): electron density, wavefunctions, restart data.

---

## Convergence Strategy

### Choosing ecutwfc

Start with the pseudopotential's suggested cutoff. Increase in steps of 5-10 Ry until total energy changes < 1 meV/atom:

| Material type | Typical ecutwfc |
|--------------|----------------|
| sp-bonded (Si, C, Al) | 20-40 Ry |
| d-electron metals (Fe, Cu) | 40-60 Ry |
| First-row elements (O, N, F) | 50-80 Ry |

### Choosing k-point Grid

For semiconductors/insulators, start with 4×4×4 and increase until energy converges to < 1 meV/atom. Metals need denser grids (8×8×8 or more) plus smearing.

### Smearing

| System | Smearing | degauss |
|--------|----------|---------|
| Insulator/semiconductor | `none` | — |
| Metal | `marzari-vanderbilt` | 0.01-0.02 Ry |
| Magnetic metal | `fermi-dirac` | 0.01 Ry |

### SCF Convergence Tips

- If SCF oscillates: reduce `energy_threshold`, try tighter density mixing
- If SCF is slow: increase ecutwfc (better starting density)
- Pulay/DIIS mixing handles most systems automatically
- Max 200 SCF steps enforced; if not converging by 100 steps, check input

---

## Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| "ecutrho must be >= 4*ecutwfc" | Explicit ecutrho too small | Set ecutrho=0 for auto |
| "Energy oscillation > 1 Ry" | SCF diverging | Reduce ecutwfc, check PP |
| "UPF parse failure" | Malformed pseudopotential | Re-download UPF file |
| "Davidson divergence" | Poor initial guess | Auto-switches to LOBPCG |
| "Negative density > 1e-6" | Numerical instability | Increase ecutwfc |
| "Unknown YAML key" | Typo in input | Check field names above |
| SCF not converging | Poor k-grid or cutoff | Increase both, add smearing for metals |
