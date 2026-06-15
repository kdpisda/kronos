# KRONOS User Guide

KRONOS (Kohn-Residual Optimized Numerics Over Silicon) is a research-grade plane-wave DFT engine for computing ground-state total energy, electronic density, Kohn-Sham eigenvalues, and ionic forces for periodic crystalline systems.

## 1. Quick Start

### Dependencies

Required: C++20 compiler (GCC 11+, Clang 14+), CMake 3.20+, FFTW3, BLAS, LAPACK, yaml-cpp.
Optional: libxc 6.0+ (built-in LDA fallback if absent), HDF5, MPI.

```bash
# Ubuntu/Debian
sudo apt install build-essential cmake libfftw3-dev libblas-dev liblapack-dev libyaml-cpp-dev

# macOS
brew install cmake fftw yaml-cpp
```

### Build and Run

```bash
cmake -B build -S .
cmake --build build -j$(nproc)
cd build && ctest --output-on-failure   # run tests

# First calculation
./build/src/kronos examples/si_bulk.yaml > si_result.json
```

GPU builds use `-DKRONOS_GPU_BACKEND=cuda`, `-DKRONOS_GPU_BACKEND=hip`, or `-DKRONOS_GPU_BACKEND=metal` (Apple Silicon, fp32 research tier -- see Section 2 note).

## 2. YAML Input File Reference

KRONOS enforces strict schema validation -- unknown keys cause a hard error. Five top-level sections are recognized: `system`, `calculation`, `convergence`, `pseudopotentials`, and `hardware`.

### Complete Example

```yaml
system:
  lattice:                        # 3x3 matrix, rows are lattice vectors (Angstrom)
    - [0.0, 2.7155, 2.7155]
    - [2.7155, 0.0, 2.7155]
    - [2.7155, 2.7155, 0.0]
  atoms:
    - {symbol: Si, position: [0.00, 0.00, 0.00]}   # fractional coordinates
    - {symbol: Si, position: [0.25, 0.25, 0.25]}

calculation:
  type: scf                       # scf | relax | bands | dos
  ecutwfc: 30.0                   # plane-wave cutoff (Ry), range 10-500
  ecutrho: 120.0                  # density cutoff (Ry), >= 4*ecutwfc; omit for auto
  kpoints: [4, 4, 4, 0, 0, 0]    # [nk1, nk2, nk3, shift1, shift2, shift3]
  xc: LDA_PZ                     # LDA_PZ | LDA_PW | PBE | PBEsol
  smearing: gaussian              # none | gaussian | marzari-vanderbilt | fermi-dirac
  degauss: 0.01                   # smearing width (Ry)
  spin: false                     # enable spin polarization

pseudopotentials:
  Si: pseudopotentials/Si.pz-vbc.UPF

convergence:
  energy: 1e-6                    # SCF energy threshold (Ry)
  density: 1e-6                   # SCF density threshold
  max_scf_steps: 100              # max iterations (hard limit: 200)
  force: 1e-3                     # force threshold for relax (Ry/bohr)

hardware:
  use_gpu: false
  gpu_backend: none               # none | cuda | hip | metal
  apple_fast_mode: false          # Apple-only: opt-in fp32 fast path (NOT validation-grade)
  mpi_tasks: 1
```

### Field Reference

| Field | Type | Default | Allowed Values | Description |
|-------|------|---------|----------------|-------------|
| `system.lattice` | 3x3 float | required | det > 0 | Lattice vectors in Angstrom (rows) |
| `system.atoms[].symbol` | string | required | element symbol | Atomic species |
| `system.atoms[].position` | 3-float | required | fractional coords | Crystal coordinates, ideally in [0,1) |
| `calculation.type` | string | `scf` | scf, relax, bands, dos | Calculation type |
| `calculation.ecutwfc` | float | required | 10 -- 500 | Wavefunction cutoff (Ry) |
| `calculation.ecutrho` | float | 4*ecutwfc | >= 4*ecutwfc | Density cutoff (Ry) |
| `calculation.kpoints` | 6-int | [1,1,1,0,0,0] | grid >= 1, shift 0/1 | Monkhorst-Pack grid + shift |
| `calculation.xc` | string | `LDA_PZ` | LDA_PZ, LDA_PW, PBE, PBEsol | XC functional |
| `calculation.smearing` | string | `none` | none, gaussian, marzari-vanderbilt, fermi-dirac | Occupation smearing |
| `calculation.degauss` | float | 0.01 | > 0 | Smearing width (Ry) |
| `calculation.spin` | bool | false | true/false | Spin polarization |
| `convergence.energy` | float | 1e-8 | > 0 | Energy tolerance (Ry) |
| `convergence.density` | float | 1e-9 | > 0 | Density tolerance |
| `convergence.max_scf_steps` | int | 100 | 1 -- 200 | Max SCF iterations |
| `convergence.force` | float | 1e-3 | > 0 | Force tolerance (Ry/bohr) |
| `pseudopotentials.<El>` | string | required | file path | Path to UPF v2 file |
| `hardware.use_gpu` | bool | false | true/false | Enable GPU |
| `hardware.gpu_backend` | string | `none` | none, cuda, hip, metal | GPU backend |
| `hardware.apple_fast_mode` | bool | false | true/false | Apple Metal only: opt-in fp32 fast path. NOT validation-grade; see note below. |
| `hardware.mpi_tasks` | int | 1 | >= 1 | MPI task count |

Every atomic species in `system.atoms` must have a matching `pseudopotentials` entry. Pseudopotential paths resolve relative to the current working directory.

**Note on `apple_fast_mode`:** Setting `hardware.apple_fast_mode: true` (or passing `--apple-fast-mode` on the CLI) enables the Apple Silicon Metal GPU path. Because Apple's Metal Shading Language has no `double` type, all GPU computations run in fp32. This is intentionally opt-in and NOT validation-grade: the validation test suite refuses to run in this mode. Use it for local development and iteration speed only; use CUDA or HIP for science-grade results. GPU builds: `-DKRONOS_GPU_BACKEND=metal` requires Xcode.app (macOS 13+) with the Metal toolchain — CLT-only installs are insufficient.

## 3. Example Calculations

### Si Bulk SCF (Semiconductor)

```yaml
system:
  lattice:
    - [0.0, 2.7155, 2.7155]
    - [2.7155, 0.0, 2.7155]
    - [2.7155, 2.7155, 0.0]
  atoms:
    - {symbol: Si, position: [0.0, 0.0, 0.0]}
    - {symbol: Si, position: [0.25, 0.25, 0.25]}
calculation:
  type: scf
  ecutwfc: 30.0
  xc: LDA_PZ
  kpoints: [4, 4, 4, 0, 0, 0]
pseudopotentials:
  Si: pseudopotentials/Si.pz-vbc.UPF
convergence:
  energy: 1e-6
  max_scf_steps: 100
```

FCC primitive vectors `a/2*(0,1,1)`, `a/2*(1,0,1)`, `a/2*(1,1,0)` with `a = 5.431 A`. Two basis atoms at (0,0,0) and (1/4,1/4,1/4). No smearing needed for this semiconductor.

### QE Validation (Si diamond, celldm(1)=10.20)

```yaml
system:
  lattice:
    - [0.0, 2.69880378, 2.69880378]
    - [2.69880378, 0.0, 2.69880378]
    - [2.69880378, 2.69880378, 0.0]
  atoms:
    - {symbol: Si, position: [0.0, 0.0, 0.0]}
    - {symbol: Si, position: [0.25, 0.25, 0.25]}
calculation:
  type: scf
  ecutwfc: 18.0
  xc: LDA_PZ
  kpoints: [4, 4, 4, 1, 1, 1]
  smearing: none
pseudopotentials:
  Si: pseudopotentials/Si.pz-vbc.UPF
convergence:
  energy: 1e-8
  max_scf_steps: 100
```

Reproduces QE example01: shifted 4x4x4 MP grid, `ecutwfc=18 Ry`. Reference total energy: -15.8445 Ry. KRONOS achieves agreement to ~0.02 Ry (gap due to symmetry reduction differences).

### Metal: Cu FCC with Smearing

```yaml
system:
  lattice:
    - [0.0, 1.805, 1.805]
    - [1.805, 0.0, 1.805]
    - [1.805, 1.805, 0.0]
  atoms:
    - {symbol: Cu, position: [0.0, 0.0, 0.0]}
calculation:
  type: scf
  ecutwfc: 50.0
  xc: PBE
  kpoints: [8, 8, 8, 1, 1, 1]
  smearing: marzari-vanderbilt
  degauss: 0.02
pseudopotentials:
  Cu: pseudopotentials/Cu.pbe-dn-kjpaw_psl.1.0.0.UPF
```

Metals require smearing and denser k-grids. Marzari-Vanderbilt ("cold") smearing converges faster than Gaussian for metallic systems.

### Geometry Relaxation

Set `type: relax` and tighten the `force` threshold:

```yaml
calculation:
  type: relax
  ecutwfc: 30.0
  xc: PBE
  kpoints: [4, 4, 4, 0, 0, 0]
convergence:
  energy: 1e-8
  force: 1e-4
  max_scf_steps: 100
```

The BFGS optimizer iterates SCF cycles until all forces fall below the threshold.

## 4. Convergence Strategy

### Choosing ecutwfc

Start with the pseudopotential's recommended cutoff and increase in 5-10 Ry steps. Stop when total energy changes by less than 1 meV/atom. Typical values: sp-bonded (Si, C): 20-40 Ry; d-electron metals (Fe, Cu): 40-60 Ry; first-row elements (O, N, F): 50-80 Ry.

### Choosing the k-point Grid

Start at 4x4x4, double each dimension until energy converges to 1 meV/atom. Metals need 8x8x8 or finer. Shifted grids (`shift=[1,1,1]`) converge faster for cubic systems by avoiding high-symmetry points.

### Smearing for Metals

| System type | Smearing | degauss |
|-------------|----------|---------|
| Insulator/semiconductor | `none` | -- |
| Simple metal | `marzari-vanderbilt` | 0.01-0.02 Ry |
| Magnetic metal | `fermi-dirac` | 0.01 Ry |

Verify insensitivity to degauss by varying it by a factor of 2. A large `smearing_energy` in the output indicates degauss is too high.

### Convergence Checklist

1. Fix k-grid, converge ecutwfc (< 1 meV/atom change per step).
2. Fix ecutwfc, converge k-grid (< 1 meV/atom change per step).
3. For metals, verify results are insensitive to degauss.
4. For relaxation, use `convergence.force <= 1e-4` Ry/bohr.
5. Compare energy components individually to catch cancellation of errors.

## 5. Pseudopotential Requirements

### Supported Format

KRONOS requires **UPF v2** (XML-based) **norm-conserving** pseudopotentials. The parser reads PP_HEADER, PP_R, PP_RAB, PP_LOCAL, PP_BETA, PP_DIJ, PP_RHOATOM, and PP_CHI sections. Fortran-style `D` exponents are handled automatically. Norm-conservation is validated on load. Ultrasoft and PAW are not yet supported (planned for v0.8).

### Where to Download

- **PseudoDojo** (http://www.pseudo-dojo.org/) -- NC pseudopotentials with stringent validation. Use the "standard" accuracy UPF set.
- **SSSP Library** (https://www.materialscloud.org/discover/sssp) -- curated sets with recommended cutoffs per element.
- **Quantum ESPRESSO** -- ships validated pseudopotentials in `pseudo/`. The `Si.pz-vbc.UPF` used in examples comes from here.

Paths in `pseudopotentials` resolve relative to the working directory. Run KRONOS from the project root or use absolute paths.

## 6. Output Format

### JSON Summary

Written to stdout. All energies in Rydberg (unless field name contains `_ev`). Forces in Ry/bohr. Eigenvalues in Ry per k-point.

```json
{
  "calculation_type": "scf",
  "converged": true,
  "scf_steps": 6,
  "total_energy_ry": -15.84388657,
  "total_energy_ev": -215.5670586,
  "fermi_energy_ev": 6.683025046,
  "energy_components": {
    "kinetic_energy": 6.217669585,
    "hartree_energy": 1.079781174,
    "xc_energy": -4.817300104,
    "local_pp_energy": -4.994074776,
    "nonlocal_pp_energy": 3.569797306,
    "ewald_energy": -16.89975976,
    "smearing_energy": 0
  },
  "crystal": { "num_atoms": 2, "volume_bohr3": 265.30 },
  "forces": [[5.78e-19, 4.52e-19, 5.74e-19], [-5.78e-19, -3.79e-19, -5.06e-19]],
  "eigenvalues": [{"kpoint_index": 0, "values_ry": [-0.4117, 0.3419, ...]}],
  "timing": {"eigensolver": 69.7, "hamiltonian_apply": 30.0, "scf_step": 73.7}
}
```

When SCF does not converge, output is still written with `"converged": false` and partial results. The `timing` section reports wall-clock seconds for the eigensolver, Hamiltonian application (FFT + nonlocal), Ewald, forces, and total SCF time. HDF5 output (density, wavefunctions, restart data) is available when compiled with HDF5 support.

## 7. Troubleshooting

### SCF Not Converging

- Increase `ecutwfc` for a better variational basis.
- Add a k-point shift or use a denser grid.
- For metals, enable smearing with appropriate `degauss`.
- KRONOS aborts if energy oscillates by > 1 Ry between steps -- check PP and lattice.
- Pulay/DIIS mixing (8-step history) is automatic; persistent oscillations signal input errors.

### Negative Electron Density

Values below zero are clamped with a warning. If magnitude exceeds 1e-6, KRONOS aborts. Fix: increase `ecutwfc` or check PP compatibility.

### Davidson Eigensolver Failure

If the Davidson residual exceeds 1e3, KRONOS auto-falls back to LOBPCG for that k-point. Subspace size is 3 * N_bands. Persistent failures indicate a poorly conditioned Hamiltonian.

### UPF Parse Errors

Ensure UPF v2 format, norm-conserving type, and that the file is not truncated. Re-download from PseudoDojo or QE if parsing fails.

### Common Error Messages

| Error | Cause | Fix |
|-------|-------|-----|
| `ecutrho must be >= 4*ecutwfc` | Explicit ecutrho too small | Remove ecutrho to auto-set, or increase it |
| `Energy oscillation > 1 Ry` | SCF diverging | Check PP and lattice vectors |
| `Unknown key 'X'` | Typo in YAML | Check field names in reference table |
| `Missing entry for element 'X'` | No PP for a species | Add element to `pseudopotentials` |
| `Negative density > 1e-6` | Numerical instability | Increase ecutwfc |
| `ecutwfc: must be in range [10, 500]` | Cutoff out of bounds | Use 10-500 Ry |
| `system.lattice: determinant must be positive` | Left-handed vectors | Reorder vectors so det > 0 |
| `Cannot open UPF file` | Wrong path | Paths resolve from working directory |
| `kpoints: must be a 6-element array` | Format error | Use `[nk1, nk2, nk3, sk1, sk2, sk3]` |
