# KRONOS Metal Backend — Design Spec

**Date:** 2026-05-16 (original) / 2026-06-15 (revised after MSL fp64 discovery)
**Author:** Kuldeep Pisda (via Claude brainstorming session)
**Status:** Revised — fp32-only Apple dev tier
**Target version:** v0.5.1 (Apple Silicon increment on top of v0.5 GPU support)

## 0. 2026-06-15 REVISION — fp64 not available on Apple GPU

The original spec (§3 Precision strategy, "option D") planned to ship a fp64
default mode on Apple GPU via Metal Shading Language's emulated `double`. That
turned out to be impossible: Apple's MSL compiler **refuses `double` outright**
("'double' is not supported in Metal" — Xcode 26.5 / Metal Toolchain v17.6).
The type is reserved (`__Reserved_Name__Do_not_use_double2`) and there is no
hardware fp64 path on any Apple GPU.

**Pivot:** the Apple Metal backend is now a fp32-only "research/dev" tier
(closer to original brainstorming option C). Implications:
- Apple GPU runs are **never validation-grade**. The < 2 meV/atom Delta test
  remains a CUDA/HIP responsibility on real fp64 hardware.
- `apple_fast_mode` is no longer "opt-in fp32"; on Apple builds it is the
  **only mode**. The flag is retained for explicit acknowledgement at runtime.
- `gpu::gemm` and `gpu::fft` on the Metal backend narrow `complex<double>` →
  `complex<float>` at the device boundary when `apple_fast_mode == true`. When
  the flag is false, the Metal backend declines the call (throws
  `GPUNotAvailableError`) so the existing `GPUHamiltonian` fallback routes it
  to CPU.
- Numerical agreement tests use fp32 tolerances (~1e-4 to 1e-5), not 1e-12.
- The validation suite refuses to run with the Metal backend active.

Sections below describe the **revised** design.

## 1. Motivation

KRONOS currently supports two GPU backends: NVIDIA (CUDA) and AMD (HIP). The
primary KRONOS development machine runs Apple Silicon (M-series), which means
the entire GPU code path in `src/gpu/` cannot be exercised locally — bugs only
surface on remote NVIDIA/AMD hardware. Adding a Metal backend enables on-device
testing and unlocks the Apple Silicon installed base as a deployment target.

Goal: **dev/test parity** with the CUDA path on Apple Silicon. Production-grade
performance on Apple GPUs is **not** a goal of this increment.

## 2. Constraints

### 2.1 Numerical hard rule
KRONOS requires `complex128` (fp64) wavefunction coefficients (see
`CLAUDE.md` § Numerical Constraints). This is non-negotiable for the default
build because the < 2 meV/atom Delta-test target depends on it.

### 2.2 Apple Silicon GPU fp64 reality
- Apple GPUs have no dedicated fp64 ALUs. `double` is supported in Metal
  Shading Language (Metal 2.4+) via software emulation, typically 10–30× slower
  than fp32.
- `MPSMatrixMultiplication`: fp32/fp16/bfloat16 only — no fp64.
- `MPSGraph FFT`: fp32/fp16 only — no fp64.
- Conclusion: vendor primitives are unusable for the default fp64 path; we
  must either write custom MSL kernels or use a portable third-party library.

### 2.3 Existing abstraction boundary
The `gpu::` namespace (`src/gpu/`) is the only place that may call vendor
APIs. Physics code must remain backend-agnostic. The Metal backend MUST live
entirely behind the existing `gpu::GPUFFTGrid`, `gpu::DeviceBuffer<T>`,
`gpu::gemm`, `gpu::gpu_malloc/free/memcpy_*`, and `gpu::GPUContext`
interfaces.

## 3. Precision strategy

Two precision modes, selectable at runtime:

| Mode | Default? | FFT | GEMM | Accuracy target | Use case |
|------|----------|-----|------|-----------------|----------|
| **fp64** | yes | VkFFT (fp64 complex) | Custom MSL kernel (`double2`) | ≤ 1e-12 vs CPU | Correctness-critical work, validation runs |
| **fp32 fast** | no — opt-in | VkFFT (fp32 complex) | Custom MSL kernel (`float2`) | ≤ 1e-5 vs CPU | Iteration speed during development; never for science |

Fast mode is gated behind:
- CLI flag: `--apple-fast-mode`
- YAML key: `hardware.apple_fast_mode: true`

When fast mode is active:
- `Logger::instance().warning("apple_fast_mode", "fp32 GPU path active — results are not validation-grade")` is emitted on startup
- Output JSON includes `"apple_fast_mode": true`
- Validation/Delta-test runners refuse to start (hard abort)

## 4. Architecture

### 4.1 Files to add

```
src/gpu/
  gpu_context_metal.cpp     # MTLDevice + MTLCommandQueue init via metal-cpp
  fft_metal.cpp             # VkFFT wrapper, both precisions
  blas_metal.cpp            # Complex GEMM dispatcher (calls .metal kernel)
  memory_metal.cpp          # MTLBuffer (storageModeShared) over gpu_malloc/free/memcpy_*
  kernels/
    complex_gemm.metal      # Templated complex GEMM kernel (fp32 + fp64)
    pointwise.metal         # Pointwise multiply, scatter, gather (future expansion)
```

No `.mm` (Objective-C++) files. metal-cpp provides pure C++17 headers so all
implementation files are plain `.cpp`.

### 4.2 Dependencies (added)

| Dependency | Version | Source | How fetched |
|-----------|---------|--------|-------------|
| metal-cpp | macOS 14 SDK release | https://developer.apple.com/metal/cpp/ | CMake `FetchContent` from Apple's tarball URL |
| VkFFT | 1.3.4 or later | https://github.com/DTolm/VkFFT | CMake `FetchContent` (git tag pinned) |

Both are header-only / single-archive distributions. No new build step
beyond what CMake already does for googletest.

Frameworks linked on Apple builds: `Metal`, `Foundation`, `QuartzCore`.

### 4.3 Build system changes

`CMakeLists.txt` (root):
```cmake
set_property(CACHE KRONOS_GPU_BACKEND PROPERTY STRINGS none cuda hip metal)
```

`src/CMakeLists.txt`: add a third branch alongside the existing `cuda`/`hip`
ones:
```cmake
elseif(KRONOS_GPU_BACKEND_LOWER STREQUAL "metal")
    if(NOT APPLE OR NOT CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
        message(FATAL_ERROR
          "KRONOS_GPU_BACKEND=metal requires Apple Silicon (macOS arm64)")
    endif()
    # FetchContent metal-cpp, VkFFT; add metal sources; link frameworks
endif()
```

`.metal` source files are compiled via Apple's `xcrun metal`/`metallib`
toolchain at build time; the resulting `.metallib` is embedded into the
binary or loaded at runtime from `$<TARGET_FILE_DIR>/kronos.metallib`.
**Implementation plan decision point:** embedded vs. side-loaded — defer to
the plan.

### 4.4 Memory model

Apple Silicon has unified memory. `storageModeShared` `MTLBuffer` is
zero-copy CPU-visible. The `gpu::gpu_memcpy_h2d` / `d2h` functions become:
- A `memcpy` between the host pointer and the buffer's `contents` pointer
  (mandatory because the host buffer is not necessarily the MTLBuffer pointer)
- Followed by `didModifyRange:` on the MTLBuffer for h2d, or a
  `waitUntilCompleted` synchronization for d2h reads of GPU-written data

This keeps the H2D/D2H interface unchanged. **No new abstraction is exposed**;
unified-memory optimization is deferred to a later increment that would
refactor the host-side wavefunction storage to be MTLBuffer-backed directly.

### 4.5 Complex GEMM kernel

One MSL kernel templated on `T` (where `T ∈ {float, double}`):

```msl
template <typename T>
kernel void zgemm(
    constant int& m, constant int& n, constant int& k,
    constant T2& alpha, constant T2& beta,
    device const T2* A [[buffer(0)]],
    device const T2* B [[buffer(1)]],
    device T2* C [[buffer(2)]],
    constant int& lda, constant int& ldb, constant int& ldc,
    uint2 gid [[thread_position_in_grid]]) { ... }
```

Two specializations compiled into the metallib: `zgemm_fp32` and `zgemm_fp64`.
Dispatcher in `blas_metal.cpp` selects by precision mode. Tiled with
threadgroup memory; tile size to be tuned to a reasonable default (16×16),
not optimized further in this increment.

### 4.6 FFT — VkFFT integration

VkFFT exposes a Metal backend (`VKFFT_BACKEND_METAL`). Wrapper in
`fft_metal.cpp` constructs a `VkFFTConfiguration` per FFT grid, holds the
`VkFFTApplication`, and exposes `forward`/`inverse` via the existing
`gpu::GPUFFTGrid` interface. Precision is determined at construction time
from the active mode (fp64 default, fp32 in fast mode).

3D mixed-radix grids (powers of 2, 3, 5) are VkFFT's bread and butter; KRONOS
FFT grids fit comfortably.

### 4.7 Error handling

Identical to CUDA path:
- MTLDevice creation failure → throw `gpu::GPUNotAvailableError`
- VkFFT init failure → throw `gpu::GPUNotAvailableError`
- Out-of-memory `MTLBuffer` allocation → throw `gpu::GPUNotAvailableError`
- All caught by `GPUHamiltonian::apply()` → CPU fallback with warning

## 5. Testing

### 5.1 TDD discipline
Per `superpowers:test-driven-development`, every new component begins with a
failing test. The order is: extend `test_gpu.cpp` first, then implement until
green.

### 5.2 New tests

1. **`GPU.MetalContextInit`** — `gpu::GPUContext::instance().init()` succeeds
   on Apple Silicon, reports a non-empty device name.
2. **`GPU.MetalMemoryRoundTrip`** — upload, `memcpy_d2d`, download via
   `DeviceBuffer<complex_t>`; bitwise equality.
3. **`GPU.MetalFFTRoundTripFP64`** — forward then inverse 3D FFT recovers
   input to within 1e-13 absolute (fp64).
4. **`GPU.MetalFFTRoundTripFP32`** — fast-mode variant, within 1e-5.
5. **`GPU.MetalComplexGEMM`** — small complex128 GEMM (m=n=k=64) matches
   reference CPU ZGEMM to 1e-12.
6. **`GPU.MetalHamiltonianApplyFP64`** — `GPUHamiltonian::apply` on a Si bulk
   minimal cell, single k-point, agrees with `Hamiltonian::apply` to 1e-12 per
   coefficient.
7. **`GPU.AppleFastModeWarning`** — enabling fast mode emits the
   `apple_fast_mode` warning event.
8. **`GPU.AppleFastModeRefusesValidation`** — running the Delta-test
   regression runner with fast mode active is a hard abort.
9. **`Integration.SiBulkOnMetal`** — full SCF for Si bulk diamond cell in fp64
   Metal mode, total energy matches CPU within 10 µRy.

### 5.3 Test gating
All Metal tests are skipped when `gpu::gpu_available() == false` (i.e.,
non-Apple builds and Apple builds where MTLDevice creation failed). This
mirrors the existing pattern in `test_gpu.cpp`.

### 5.4 Regression invariant
The full existing 440-test suite continues to pass with
`KRONOS_GPU_BACKEND=none`. No CPU-path regressions allowed.

## 6. Out of scope (explicitly)

- Full device-resident H|ψ⟩ pipeline. The current CUDA `apply_gpu` already
  contains host roundtrips (scatter/gather + V_eff multiply); we match that
  state on Apple, not exceed it. End-to-end GPU residency is a separate later
  cross-backend optimization.
- Performance tuning (kernel auto-tuning, threadgroup size sweeps, MPSGraph
  for fp32). Correctness over speed.
- Unified-memory refactor (host wavefunctions backed by `MTLBuffer`). Would
  break the host-only CPU path; out of scope.
- Production validation suite on Apple GPU. fp64 emulation is too slow for
  full Delta-test runs; that remains a CUDA/HIP responsibility.
- Intel Mac support. Apple Silicon arm64 only.

## 7. Acceptance criteria

1. `cmake -B build -S . -DKRONOS_GPU_BACKEND=metal` succeeds on an M-series Mac.
2. `ctest -j2` from that build passes with zero failures (skips allowed for
   non-applicable tests, e.g., CUDA-only).
3. `ctest` from a CPU-only build (`-DKRONOS_GPU_BACKEND=none`) still reports
   440/440 passed — no regressions.
4. The new GPU-active tests (§ 5.2 items 1–9) all pass on Apple Silicon.
5. `Integration.SiBulkOnMetal` SCF total energy is within 10 µRy of the CPU
   reference (well below the < 2 meV/atom target).
6. Documentation updated: `README.md`, `docs/user_guide.md`,
   `docs/architecture.md`, `CLAUDE.md` — all reference the new `metal`
   backend option and the fp64-performance caveat.
7. CHANGELOG / release notes entry for v0.5.1.

## 8. Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| VkFFT fp64 Metal path has bugs we hit at small grid sizes | Med | High | Pin VkFFT version; isolate via wrapper; FFT roundtrip test catches it early |
| metal-cpp `FetchContent` URL changes / breaks | Low | Med | Pin to a specific tarball; fallback to vendored copy in `external/` if needed |
| Apple fp64 emulation is so slow that `Integration.SiBulkOnMetal` exceeds ctest timeout | Med | Med | Use smallest possible Si cell (2 atoms, 2×2×2 k-grid, low ecutwfc); accept multi-minute runtime; mark `LARGE` |
| Custom complex GEMM has subtle indexing bug at non-tile-aligned sizes | Med | High | Test at sizes 1, 13, 16, 17, 64, 65 — boundary coverage |
| MSL `double` support varies by macOS version | Low | High | Require macOS 13+ at CMake configure; document in README |

## 9. Future increments (post-v0.5.1)

- v0.5.2: Full device-resident H|ψ⟩ pipeline (custom scatter/gather/pointwise
  kernels) — applies to all three backends.
- v0.5.3: MPSGraph fast path for fp32 mode if benchmarks justify the second
  code path.
- v0.6: Apple GPU Davidson eigensolver (currently host-side).
- Eventual: unified-memory refactor — host wavefunction storage backed by
  `MTLBuffer` directly, eliminating H2D/D2H entirely on Apple.

## 10. Open questions for the implementation plan

1. `.metallib` build integration: embed via `xxd -i` blob into binary, or
   side-load from same dir as executable? (Side-load is simpler; embed makes
   the binary self-contained.)
2. VkFFT pin: latest stable tag, or a tag known to work with Metal? (Resolve
   when writing the plan.)
3. Tile size for complex GEMM kernel: default 16×16 is fine for shipping; do
   we sanity-check on M3 Pro at all, or accept any non-pathological perf?
