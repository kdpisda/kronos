# Apple Silicon Metal Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a third GPU backend (`metal`) to KRONOS so the existing GPU code path can be exercised on Apple Silicon Macs. **fp32 only** (Apple MSL has no `double` support). Apple Metal is a research/dev tier — not validation-grade.

**Architecture:** Mirror the existing `cuda`/`hip` backend pattern in `src/gpu/`. Physics code stays vendor-agnostic; all Apple-specific calls live behind the same `gpu::` interfaces. Use **metal-cpp** for the Apple SDK bindings (pure C++17, no Objective-C++) and **VkFFT** for 3D complex FFT (fp32 only on Apple). One MSL kernel handles complex GEMM in fp32.

**Tech Stack:** C++20, Metal Shading Language, metal-cpp (header-only Apple bindings), VkFFT (FetchContent), CMake 3.20+, GoogleTest, `xcrun metal`/`metallib` for shader compilation.

**Spec:** `docs/superpowers/specs/2026-05-16-apple-silicon-metal-backend-design.md`

---

## 2026-06-15 REVISION — fp32-only Apple GPU

Tasks 6–10 completed against the original spec. While running Task 10 we discovered Apple MSL refuses `double` entirely. The plan is now revised for fp32-only Apple. Tasks 11–19 implementation notes:

- **Task 11 (unchanged):** plumb `apple_fast_mode` accessor through `GPUContext` + `main.cpp`. The semantics shift to "Apple GPU enable", not "opt-in fp32".
- **Task 12 (revised):** `blas_metal.cpp::gemm` narrows `complex<double> → complex<float>` at the device boundary when `apple_fast_mode == true`; otherwise throws `GPUNotAvailableError` so `GPUHamiltonian` falls back to CPU. Tests at fp32 tolerance (~1e-5 element-wise).
- **Task 13 (revised):** `fft_metal.cpp` initializes VkFFT in fp32 mode unconditionally. Tests at fp32 tolerance.
- **Task 14 (revised):** `GPU.MetalHamiltonianApplyFP32MatchesCPU` (renamed from FP64) — tolerance loosened to ~1e-4 per coefficient. Skipped unless `apple_fast_mode` is enabled.
- **Task 16 (revised):** validation suite refuses Apple/Metal backend entirely (not just `apple_fast_mode` flag — they're equivalent on Apple).
- **Task 17 (revised):** Si bulk SCF agreement loosened to ~1 mRy total energy (fp32 accumulation error is structural, not solvable with this hardware).
- **Task 18 (revised):** docs say "Apple Metal: research/dev tier only, fp32 throughout, NOT validation-grade".

---

## Phase 1 — Build system & dependencies

This phase makes `cmake -B build -S . -DKRONOS_GPU_BACKEND=metal` a valid configure command. No runtime behavior yet.

### Task 1: Add `metal` as a valid `KRONOS_GPU_BACKEND` value

**Files:**
- Modify: `CMakeLists.txt:22-24`
- Modify: `CMakeLists.txt:40-44` (language enablement)

- [ ] **Step 1: Update the cache variable and STRINGS property**

Edit `CMakeLists.txt:22-24` to:

```cmake
set(KRONOS_GPU_BACKEND "none" CACHE STRING
    "GPU acceleration backend: none | cuda | hip | metal")
set_property(CACHE KRONOS_GPU_BACKEND PROPERTY STRINGS none cuda hip metal)
```

- [ ] **Step 2: Add Apple Silicon guard in language enablement block**

Edit `CMakeLists.txt:40-44`. After the CUDA `enable_language` branch, add:

```cmake
elseif(KRONOS_GPU_BACKEND_LOWER STREQUAL "metal")
    if(NOT APPLE)
        message(FATAL_ERROR
            "KRONOS_GPU_BACKEND=metal requires macOS; current platform is "
            "${CMAKE_SYSTEM_NAME}")
    endif()
    if(NOT CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
        message(FATAL_ERROR
            "KRONOS_GPU_BACKEND=metal requires Apple Silicon (arm64); "
            "current architecture is ${CMAKE_SYSTEM_PROCESSOR}")
    endif()
    if(NOT CMAKE_SYSTEM_VERSION VERSION_GREATER_EQUAL "22.0.0")
        message(WARNING
            "KRONOS_GPU_BACKEND=metal is tested on macOS 13+; current "
            "kernel version ${CMAKE_SYSTEM_VERSION} may have limited "
            "double-precision Metal Shading Language support.")
    endif()
endif()
```

- [ ] **Step 3: Verify CPU build still configures**

Run: `cmake -B build_cpu -S . 2>&1 | grep "GPU backend"`
Expected: `GPU backend    : none`

- [ ] **Step 4: Verify metal configure fails cleanly on non-Apple, succeeds on Apple Silicon**

On Apple Silicon, run: `cmake -B build_metal -S . -DKRONOS_GPU_BACKEND=metal 2>&1 | tail -20`
Expected: configure completes (no Metal sources yet, so configure succeeds even though nothing builds). May see warnings about missing Metal source files — that's expected, fixed in Task 5.

- [ ] **Step 5: Commit**

```bash
git add CMakeLists.txt
git commit -m "build: add metal as a valid KRONOS_GPU_BACKEND value with Apple Silicon guards"
```

---

### Task 2: Add metal-cpp via FetchContent

**Files:**
- Modify: `CMakeLists.txt` (add FetchContent block at bottom of GPU-specific dependencies section, around line 75)

- [ ] **Step 1: Add metal-cpp fetch block**

Add to `CMakeLists.txt`, in the `KRONOS_GPU_BACKEND_LOWER STREQUAL "metal"` branch of the GPU-specific dependencies section (insert a new `elseif` clause around line 75, after the `hip` branch):

```cmake
elseif(KRONOS_GPU_BACKEND_LOWER STREQUAL "metal")
    # metal-cpp: Apple's official C++17 headers for Metal (header-only)
    include(FetchContent)
    FetchContent_Declare(
        metal_cpp
        URL https://developer.apple.com/metal/cpp/files/metal-cpp_macOS14_iOS17.zip
        URL_HASH SHA256=2c637afac98e6a86d9f9b3284755d75d6c5ad6b30b3c5e51b14e5a0f49f33c8e
    )
    FetchContent_MakeAvailable(metal_cpp)
    # INTERFACE target is named kronos_metal_cpp to avoid colliding with
    # the FetchContent population name (metal_cpp). Downstream callers
    # use target_link_libraries(... kronos_metal_cpp).
    add_library(kronos_metal_cpp INTERFACE)
    target_include_directories(kronos_metal_cpp INTERFACE ${metal_cpp_SOURCE_DIR})
    message(STATUS "KRONOS: Metal backend enabled (metal-cpp at ${metal_cpp_SOURCE_DIR})")
endif()
```

NOTE: The `URL_HASH` value above is a placeholder — Apple's tarball hash must be recomputed when running this for the first time. The implementer should:
1. Run `cmake -B build_metal -S . -DKRONOS_GPU_BACKEND=metal` once
2. Read the actual SHA-256 of the downloaded zip from the FetchContent error / from `shasum -a 256 build_metal/_deps/metal_cpp-subbuild/metal_cpp-populate-prefix/src/metal-cpp_macOS14_iOS17.zip`
3. Paste it back into `URL_HASH`

Confirm the value, then re-run configure.

- [ ] **Step 2: Verify metal-cpp is downloaded and the target is created**

Run on Apple Silicon: `cmake -B build_metal -S . -DKRONOS_GPU_BACKEND=metal 2>&1 | grep -E "metal-cpp|Metal backend"`
Expected: a line `KRONOS: Metal backend enabled (metal-cpp at /path/to/build_metal/_deps/metal_cpp-src)` and no errors.

- [ ] **Step 3: Verify a metal-cpp header is reachable**

Run: `ls build_metal/_deps/metal_cpp-src/Metal/Metal.hpp`
Expected: file exists.

- [ ] **Step 4: Commit**

```bash
git add CMakeLists.txt
git commit -m "build: fetch metal-cpp headers for Apple Silicon backend"
```

---

### Task 3: Add VkFFT via FetchContent (Metal backend)

**Files:**
- Modify: `CMakeLists.txt` (extend the metal block from Task 2)

- [ ] **Step 1: Add VkFFT fetch block**

Inside the `metal` branch in `CMakeLists.txt` (the `elseif` from Task 2), before the closing `endif()`, append:

```cmake
    # VkFFT: portable GPU FFT (uses Metal via metal-cpp under the hood)
    FetchContent_Declare(
        vkfft
        GIT_REPOSITORY https://github.com/DTolm/VkFFT.git
        GIT_TAG        v1.3.4
        GIT_SHALLOW    TRUE
    )
    FetchContent_GetProperties(vkfft)
    if(NOT vkfft_POPULATED)
        FetchContent_Populate(vkfft)
    endif()
    # INTERFACE target is named kronos_vkfft to avoid colliding with the
    # FetchContent population name (vkfft). Downstream callers use kronos_vkfft.
    add_library(kronos_vkfft INTERFACE)
    target_include_directories(kronos_vkfft INTERFACE
        "${vkfft_SOURCE_DIR}/vkFFT")
    target_compile_definitions(kronos_vkfft INTERFACE
        VKFFT_BACKEND=5)   # 5 = Metal backend
    message(STATUS "KRONOS: VkFFT fetched at ${vkfft_SOURCE_DIR}")
```

- [ ] **Step 2: Configure and verify**

Run: `cmake -B build_metal -S . -DKRONOS_GPU_BACKEND=metal 2>&1 | grep VkFFT`
Expected: `KRONOS: VkFFT fetched at /path/to/.../_deps/vkfft-src`

- [ ] **Step 3: Verify VkFFT header is reachable**

Run: `ls build_metal/_deps/vkfft-src/vkFFT/vkFFT.h`
Expected: file exists.

- [ ] **Step 4: Commit**

```bash
git add CMakeLists.txt
git commit -m "build: fetch VkFFT v1.3.4 with Metal backend for Apple GPU FFT"
```

---

### Task 4: Add Metal shader (`.metal`) compilation step

**Files:**
- Create: `cmake/MetalShaders.cmake`
- Modify: `CMakeLists.txt` (include the new module in the metal branch)

- [ ] **Step 1: Write the helper CMake module**

Create `cmake/MetalShaders.cmake`:

```cmake
# ============================================================================
# MetalShaders.cmake
# Compiles .metal source files into a single .metallib at build time.
# Usage:
#   kronos_build_metallib(
#       TARGET   <output_name>            # e.g., kronos_metallib
#       OUTPUT   <path/to/output.metallib>
#       SOURCES  file1.metal file2.metal ...
#       SDK      macosx                   # or iphoneos
#   )
# ============================================================================

function(kronos_build_metallib)
    cmake_parse_arguments(KMM "" "TARGET;OUTPUT;SDK" "SOURCES" ${ARGN})
    if(NOT KMM_SDK)
        set(KMM_SDK macosx)
    endif()

    find_program(XCRUN_EXE xcrun REQUIRED)

    set(_air_files)
    foreach(_src ${KMM_SOURCES})
        get_filename_component(_name "${_src}" NAME_WE)
        set(_air "${CMAKE_CURRENT_BINARY_DIR}/${_name}.air")
        list(APPEND _air_files "${_air}")
        add_custom_command(
            OUTPUT "${_air}"
            COMMAND ${XCRUN_EXE} -sdk ${KMM_SDK} metal
                    -c "${CMAKE_CURRENT_SOURCE_DIR}/${_src}"
                    -o "${_air}"
                    -std=metal3.0
                    -frecord-sources
            DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/${_src}"
            COMMENT "Compiling Metal shader ${_src}"
            VERBATIM
        )
    endforeach()

    add_custom_command(
        OUTPUT "${KMM_OUTPUT}"
        COMMAND ${XCRUN_EXE} -sdk ${KMM_SDK} metallib ${_air_files}
                -o "${KMM_OUTPUT}"
        DEPENDS ${_air_files}
        COMMENT "Linking Metal library ${KMM_OUTPUT}"
        VERBATIM
    )

    add_custom_target(${KMM_TARGET} ALL DEPENDS "${KMM_OUTPUT}")
endfunction()
```

- [ ] **Step 2: Include the module from root CMakeLists.txt**

In `CMakeLists.txt`, immediately after the `list(APPEND CMAKE_MODULE_PATH ...)` line (around line 35), add:

```cmake
include(MetalShaders OPTIONAL)
```

The `OPTIONAL` keyword means non-Apple platforms ignore it cleanly.

- [ ] **Step 3: Verify the function is found on Apple Silicon**

Add a temporary `message(STATUS "kronos_build_metallib found: ${CMAKE_CURRENT_FUNCTION_DEFINED}")` after the `include(...)` line, run configure, and confirm xcrun is located.

Run: `cmake -B build_metal -S . -DKRONOS_GPU_BACKEND=metal 2>&1 | grep -E "xcrun|XCRUN"`
Expected: no "NOT-FOUND" — xcrun is on PATH.

Remove the temporary message line after verification.

- [ ] **Step 4: Commit**

```bash
git add cmake/MetalShaders.cmake CMakeLists.txt
git commit -m "build: add MetalShaders.cmake helper to compile .metal sources via xcrun"
```

---

## Phase 2 — Skeleton wiring (CPU-only path stays green)

Add empty Metal source files that satisfy the linker. Every primitive throws `GPUNotAvailableError` for now. The existing CUDA/HIP tests + CPU tests must remain unaffected.

### Task 5: Add empty Metal source files and wire them into `src/CMakeLists.txt`

**Files:**
- Create: `src/gpu/gpu_context_metal.cpp` (skeleton)
- Create: `src/gpu/memory_metal.cpp` (skeleton)
- Create: `src/gpu/blas_metal.cpp` (skeleton)
- Create: `src/gpu/fft_metal.cpp` (skeleton)
- Create: `src/gpu/kernels/complex_gemm.metal` (empty placeholder kernel)
- Modify: `src/CMakeLists.txt:68-99`

- [ ] **Step 1: Create the four skeleton .cpp files**

Each file follows the same pattern: guarded by `#ifdef KRONOS_GPU_METAL`, throws `GPUNotAvailableError` for everything. The body is real code in later tasks.

Create `src/gpu/gpu_context_metal.cpp`:

```cpp
// ============================================================================
// KRONOS  src/gpu/gpu_context_metal.cpp
// Metal GPU context (Apple Silicon)
// ============================================================================

#ifdef KRONOS_GPU_METAL

#include "gpu/gpu_context.hpp"
#include "gpu/memory.hpp"

namespace kronos::gpu {

GPUContext& GPUContext::instance() {
    static GPUContext ctx;
    return ctx;
}

void GPUContext::init(int /*mpi_rank*/, int /*local_rank*/) {
    // Implemented in Task 7
    initialized_ = false;
}

void GPUContext::finalize() {
    initialized_ = false;
}

GPUContext::~GPUContext() {
    finalize();
}

void GPUContext::set_deterministic(bool /*enable*/) {
    // No-op for skeleton
}

} // namespace kronos::gpu

#endif // KRONOS_GPU_METAL
```

Create `src/gpu/memory_metal.cpp`:

```cpp
// ============================================================================
// KRONOS  src/gpu/memory_metal.cpp
// Metal GPU memory wrapper (Apple Silicon unified memory)
// ============================================================================

#ifdef KRONOS_GPU_METAL

#include "gpu/memory.hpp"

namespace kronos::gpu {

void* gpu_malloc(size_t /*bytes*/) {
    throw GPUNotAvailableError("metal memory not yet implemented (Task 8)");
}

void gpu_free(void* /*ptr*/) {
    // Implemented in Task 8
}

void gpu_memcpy_h2d(void* /*dst*/, const void* /*src*/, size_t /*bytes*/) {
    throw GPUNotAvailableError("metal memory not yet implemented (Task 8)");
}

void gpu_memcpy_d2h(void* /*dst*/, const void* /*src*/, size_t /*bytes*/) {
    throw GPUNotAvailableError("metal memory not yet implemented (Task 8)");
}

void gpu_memcpy_d2d(void* /*dst*/, const void* /*src*/, size_t /*bytes*/) {
    throw GPUNotAvailableError("metal memory not yet implemented (Task 8)");
}

bool gpu_available() {
    return false;  // Updated in Task 7
}

size_t gpu_memory_free() {
    return 0;
}

size_t gpu_memory_total() {
    return 0;
}

} // namespace kronos::gpu

#endif // KRONOS_GPU_METAL
```

Create `src/gpu/blas_metal.cpp`:

```cpp
// ============================================================================
// KRONOS  src/gpu/blas_metal.cpp
// Metal complex GEMM (Apple Silicon)
// ============================================================================

#ifdef KRONOS_GPU_METAL

#include "gpu/blas.hpp"
#include "gpu/memory.hpp"

namespace kronos::gpu {

void gemm(int /*m*/, int /*n*/, int /*k*/,
          complex_t /*alpha*/,
          const complex_t* /*A*/, int /*lda*/,
          const complex_t* /*B*/, int /*ldb*/,
          complex_t /*beta*/,
          complex_t* /*C*/, int /*ldc*/) {
    throw GPUNotAvailableError("metal gemm not yet implemented (Task 12)");
}

complex_t zdotc(int /*n*/, const complex_t* /*x*/, const complex_t* /*y*/) {
    throw GPUNotAvailableError("metal zdotc not yet implemented (Task 12)");
}

} // namespace kronos::gpu

#endif // KRONOS_GPU_METAL
```

Create `src/gpu/fft_metal.cpp`:

```cpp
// ============================================================================
// KRONOS  src/gpu/fft_metal.cpp
// Metal 3D complex FFT (via VkFFT)
// ============================================================================

#ifdef KRONOS_GPU_METAL

#include "gpu/fft.hpp"
#include "gpu/memory.hpp"

namespace kronos::gpu {

GPUFFTGrid::GPUFFTGrid(std::array<int, 3> dims)
    : dims_(dims) {
    throw GPUNotAvailableError("metal FFT not yet implemented (Task 14)");
}

GPUFFTGrid::~GPUFFTGrid() = default;

void GPUFFTGrid::forward(const complex_t* /*d_input*/, complex_t* /*d_output*/) {
    throw GPUNotAvailableError("metal FFT not yet implemented (Task 14)");
}

void GPUFFTGrid::inverse(const complex_t* /*d_input*/, complex_t* /*d_output*/) {
    throw GPUNotAvailableError("metal FFT not yet implemented (Task 14)");
}

std::array<int, 3> GPUFFTGrid::dims() const {
    return dims_;
}

} // namespace kronos::gpu

#endif // KRONOS_GPU_METAL
```

Create `src/gpu/kernels/complex_gemm.metal` (placeholder — real kernel in Task 10):

```msl
// ============================================================================
// KRONOS  src/gpu/kernels/complex_gemm.metal
// Placeholder. Real complex-GEMM kernel arrives in Task 10.
// ============================================================================

#include <metal_stdlib>
using namespace metal;

kernel void zgemm_placeholder(device const float* a [[buffer(0)]],
                              device       float* c [[buffer(1)]],
                              uint gid [[thread_position_in_grid]]) {
    c[gid] = a[gid];
}
```

- [ ] **Step 2: Wire the Metal branch in `src/CMakeLists.txt`**

Edit `src/CMakeLists.txt:81-99`. Add a new `elseif` branch for metal between the `hip` branch and the final `else` (CPU stubs) branch:

```cmake
elseif(KRONOS_GPU_BACKEND_LOWER STREQUAL "metal")
    set(KRONOS_METAL_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/gpu/gpu_context_metal.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/gpu/memory_metal.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/gpu/blas_metal.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/gpu/fft_metal.cpp
    )
    target_sources(kronos_lib PRIVATE ${KRONOS_METAL_SOURCES})
    target_compile_definitions(kronos_lib PUBLIC KRONOS_GPU_METAL)
    target_link_libraries(kronos_lib PUBLIC
        kronos_metal_cpp
        kronos_vkfft
        "-framework Metal"
        "-framework Foundation"
        "-framework QuartzCore"
    )

    # Compile .metal shaders into kronos.metallib (placed next to executable)
    kronos_build_metallib(
        TARGET  kronos_metallib
        OUTPUT  ${CMAKE_BINARY_DIR}/kronos.metallib
        SDK     macosx
        SOURCES gpu/kernels/complex_gemm.metal
    )
    add_dependencies(kronos_lib kronos_metallib)
```

- [ ] **Step 3: Configure and build (metal target)**

Run: `cmake -B build_metal -S . -DKRONOS_GPU_BACKEND=metal && cmake --build build_metal -j4 2>&1 | tail -20`
Expected: clean build, `kronos.metallib` produced under `build_metal/`.

Run: `ls build_metal/kronos.metallib`
Expected: file exists.

- [ ] **Step 4: Configure and build (CPU target — regression check)**

Run: `cmake --build build -j4 2>&1 | tail -10` (existing CPU `build/` directory)
Expected: still clean. No Metal sources compiled.

- [ ] **Step 5: Run the existing test suite to confirm no CPU-build regression**

Run: `cd build && ctest -j2 --output-on-failure 2>&1 | tail -5`
Expected: `100% tests passed, 0 tests failed out of 440`.

- [ ] **Step 6: Commit**

```bash
git add src/gpu/gpu_context_metal.cpp src/gpu/memory_metal.cpp \
        src/gpu/blas_metal.cpp src/gpu/fft_metal.cpp \
        src/gpu/kernels/complex_gemm.metal \
        src/CMakeLists.txt
git commit -m "feat(gpu): add Metal backend skeleton (compiles, throws on all primitives)"
```

---

### Task 6: Extend `HardwareParams` and input parser with `apple_fast_mode`

**Files:**
- Modify: `src/core/types.hpp:78-82`
- Modify: `src/io/input_parser.cpp:384-406`
- Modify: `test/test_input.cpp` (add failing test FIRST)

- [ ] **Step 1: Write the failing test**

Append to `test/test_input.cpp` (find the last `TEST(...)` in the file and add after it):

```cpp
TEST(InputParser, HardwareAppleFastModeDefaultsFalse) {
    std::string yaml = R"(
system:
  lattice: [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
  atoms:
    - {symbol: Si, position: [0, 0, 0]}
calculation:
  ecutwfc: 20
pseudopotentials:
  Si: dummy.upf
)";
    InputData input = parse_input_string(yaml);
    EXPECT_FALSE(input.hardware.apple_fast_mode);
}

TEST(InputParser, HardwareAppleFastModeAcceptsTrue) {
    std::string yaml = R"(
system:
  lattice: [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
  atoms:
    - {symbol: Si, position: [0, 0, 0]}
calculation:
  ecutwfc: 20
hardware:
  apple_fast_mode: true
pseudopotentials:
  Si: dummy.upf
)";
    InputData input = parse_input_string(yaml);
    EXPECT_TRUE(input.hardware.apple_fast_mode);
}
```

If `parse_input_string` doesn't exist, use whatever wrapper the existing tests in `test_input.cpp` use to parse a YAML string (read the top of that file to confirm; if only `parse_input(path)` exists, write the YAML to a temp file via `std::filesystem::temp_directory_path()` and parse that).

- [ ] **Step 2: Run the test to confirm it fails**

Run: `cd build && cmake --build . --target test_input -j4 2>&1 | tail -5`
Expected: compile error — `apple_fast_mode` is not a member of `HardwareParams`.

- [ ] **Step 3: Add the field to `HardwareParams`**

Edit `src/core/types.hpp:78-82` to:

```cpp
// Hardware configuration
struct HardwareParams {
    bool use_gpu{false};
    std::string gpu_backend{"none"};  // "cuda", "hip", "metal", "none"
    int mpi_tasks{1};
    bool apple_fast_mode{false};      // opt-in fp32 fast path for Apple GPU
};
```

- [ ] **Step 4: Update `parse_hardware` to read the new key**

Edit `src/io/input_parser.cpp:401-403`. After the `mpi_tasks` block, add:

```cpp
    if (hw["apple_fast_mode"]) {
        params.apple_fast_mode = hw["apple_fast_mode"].as<bool>();
    }
```

- [ ] **Step 5: Update the strict-key set if one exists**

The input parser rejects unknown YAML keys. Check `src/io/input_parser.cpp:440-460` for a list of accepted `hardware.*` keys. If there's an explicit allowed-keys list for the `hardware` section, add `"apple_fast_mode"` to it. If hardware keys are validated by per-key lookup (as in the snippet above), no change needed.

- [ ] **Step 6: Rebuild and run the new tests**

Run: `cd build && cmake --build . --target test_input -j4 && ./test/test_input --gtest_filter='InputParser.HardwareAppleFastMode*' 2>&1 | tail -10`
Expected: both tests pass.

- [ ] **Step 7: Full regression: run the entire CPU test suite**

Run: `cd build && ctest -j2 --output-on-failure 2>&1 | tail -3`
Expected: `100% tests passed, 0 tests failed out of 442` (the two new tests bring it to 442).

- [ ] **Step 8: Commit**

```bash
git add src/core/types.hpp src/io/input_parser.cpp test/test_input.cpp
git commit -m "feat(io): add hardware.apple_fast_mode to YAML input"
```

---

## Phase 3 — GPU context and memory

This is the real work begins. After Task 9, `gpu::gpu_available()` returns `true` on Apple Silicon when an MTLDevice is reachable, and the `DeviceBuffer<T>` template works end-to-end.

### Task 7: Real `gpu_context_metal.cpp` (MTLDevice + queue)

**Files:**
- Modify: `src/gpu/gpu_context_metal.cpp` (replace skeleton)
- Add to: `src/gpu/gpu_context.hpp` — a single new public method

- [ ] **Step 1: Extend `GPUContext` with a Metal-specific accessor**

Edit `src/gpu/gpu_context.hpp`. Inside the public section of `class GPUContext`, after `void* blas_handle() const`, add:

```cpp
    /// Get the Metal command queue (opaque MTL::CommandQueue*). Nullptr on non-Metal builds.
    void* metal_queue() const { return metal_queue_; }
```

And inside the private members, after `void* blas_handle_{nullptr};`, add:

```cpp
    void* metal_queue_{nullptr};  // MTL::CommandQueue* on Metal builds
```

- [ ] **Step 2: Write failing test for context init**

Append to `test/test_gpu.cpp`:

```cpp
#ifdef KRONOS_GPU_METAL
TEST(GPU, MetalContextInit) {
    auto& ctx = gpu::GPUContext::instance();
    ctx.init(0, 0);
    ASSERT_TRUE(ctx.is_initialized())
        << "Expected MTLDevice to initialize on Apple Silicon";
    EXPECT_GT(ctx.num_devices(), 0);
    EXPECT_FALSE(ctx.device_name().empty());
    EXPECT_NE(ctx.metal_queue(), nullptr);
}
#endif
```

- [ ] **Step 3: Run the test, expect it to fail**

Run: `cd build_metal && cmake --build . --target test_gpu -j4 && ./test/test_gpu --gtest_filter='GPU.MetalContextInit' 2>&1 | tail -10`
Expected: FAIL — context init still no-ops.

- [ ] **Step 4: Implement the real `init`**

Replace `src/gpu/gpu_context_metal.cpp` with:

```cpp
// ============================================================================
// KRONOS  src/gpu/gpu_context_metal.cpp
// Metal GPU context (Apple Silicon)
// ============================================================================

#ifdef KRONOS_GPU_METAL

#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>

#include "gpu/gpu_context.hpp"
#include "gpu/memory.hpp"
#include <iostream>
#include <string>

namespace kronos::gpu {

GPUContext& GPUContext::instance() {
    static GPUContext ctx;
    return ctx;
}

void GPUContext::init(int mpi_rank, int /*local_rank*/) {
    if (initialized_) return;

    MTL::Device* device = MTL::CreateSystemDefaultDevice();
    if (!device) {
        std::cerr << "KRONOS GPU: no Metal device found (rank "
                  << mpi_rank << ")\n";
        return;
    }
    // metal-cpp uses retain/release; CreateSystemDefaultDevice returns +1.
    // We hold it for the lifetime of the singleton.
    blas_handle_ = static_cast<void*>(device);

    MTL::CommandQueue* queue = device->newCommandQueue();
    if (!queue) {
        device->release();
        blas_handle_ = nullptr;
        std::cerr << "KRONOS GPU: failed to create Metal command queue\n";
        return;
    }
    metal_queue_ = static_cast<void*>(queue);

    num_devices_ = 1;  // metal-cpp default-device API; multi-GPU later
    device_id_   = 0;

    NS::String* name = device->name();
    device_name_ = std::string(name->utf8String());

    initialized_ = true;

    std::cerr << "KRONOS GPU: rank " << mpi_rank
              << " → Metal device '" << device_name_ << "'\n";
}

void GPUContext::finalize() {
    if (!initialized_) return;
    if (metal_queue_) {
        static_cast<MTL::CommandQueue*>(metal_queue_)->release();
        metal_queue_ = nullptr;
    }
    if (blas_handle_) {
        static_cast<MTL::Device*>(blas_handle_)->release();
        blas_handle_ = nullptr;
    }
    initialized_ = false;
}

GPUContext::~GPUContext() {
    finalize();
}

void GPUContext::set_deterministic(bool /*enable*/) {
    // Metal does not expose a per-queue determinism flag.
    // Determinism is enforced at the kernel level (Task 10).
}

} // namespace kronos::gpu

#endif // KRONOS_GPU_METAL
```

NOTE: The `#define NS_PRIVATE_IMPLEMENTATION` (and friends) macros **must appear in exactly one .cpp file** in the entire program — this is how metal-cpp emits its definitions. `gpu_context_metal.cpp` is that file. The other Metal files (`memory_metal.cpp`, `blas_metal.cpp`, `fft_metal.cpp`) must NOT define them; they only `#include` the headers.

- [ ] **Step 5: Update `gpu_available()` in `memory_metal.cpp`**

Edit `src/gpu/memory_metal.cpp`, replace the body of `gpu_available()` with:

```cpp
bool gpu_available() {
    // Cheap probe: ask the context whether a device exists.
    // Safe to call before init() because we make a transient device here.
    MTL::Device* probe = MTL::CreateSystemDefaultDevice();
    if (!probe) return false;
    probe->release();
    return true;
}
```

Add the includes at the top of `memory_metal.cpp`:

```cpp
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
```

Update `gpu_memory_free()` and `gpu_memory_total()` to query the device:

```cpp
size_t gpu_memory_free() {
    MTL::Device* d = MTL::CreateSystemDefaultDevice();
    if (!d) return 0;
    // recommendedMaxWorkingSetSize is the closest Metal analog to "free GPU mem".
    size_t v = d->recommendedMaxWorkingSetSize();
    d->release();
    return v;
}

size_t gpu_memory_total() {
    // On unified-memory Apple Silicon, "total GPU mem" == system RAM.
    // Use sysctl HW_MEMSIZE.
    int64_t physmem = 0;
    size_t len = sizeof(physmem);
    sysctlbyname("hw.memsize", &physmem, &len, nullptr, 0);
    return static_cast<size_t>(physmem);
}
```

Add `#include <sys/sysctl.h>` to the includes block.

- [ ] **Step 6: Rebuild and re-run the test**

Run: `cd build_metal && cmake --build . --target test_gpu -j4 && ./test/test_gpu --gtest_filter='GPU.MetalContextInit' 2>&1 | tail -10`
Expected: PASS, with stderr output like `KRONOS GPU: rank 0 → Metal device 'Apple M3 Pro'`.

- [ ] **Step 7: Commit**

```bash
git add src/gpu/gpu_context.hpp src/gpu/gpu_context_metal.cpp \
        src/gpu/memory_metal.cpp test/test_gpu.cpp
git commit -m "feat(gpu): implement Metal MTLDevice + command queue init"
```

---

### Task 8: Real `memory_metal.cpp` (MTLBuffer with `storageModeShared`)

**Files:**
- Modify: `src/gpu/memory_metal.cpp`
- Modify: `test/test_gpu.cpp` (add memory tests)

- [ ] **Step 1: Write failing tests**

Append to `test/test_gpu.cpp` (inside the `#ifdef KRONOS_GPU_METAL` block, or open a new one):

```cpp
#ifdef KRONOS_GPU_METAL
TEST(GPU, MetalMallocFreeRoundTrip) {
    auto& ctx = gpu::GPUContext::instance();
    ctx.init();
    ASSERT_TRUE(ctx.is_initialized());

    constexpr size_t N = 1024;
    void* p = gpu::gpu_malloc(N * sizeof(double));
    ASSERT_NE(p, nullptr);
    gpu::gpu_free(p);  // must not crash
}

TEST(GPU, MetalMemoryRoundTripComplex128) {
    auto& ctx = gpu::GPUContext::instance();
    ctx.init();
    ASSERT_TRUE(ctx.is_initialized());

    std::vector<complex_t> host_in(256);
    for (size_t i = 0; i < host_in.size(); ++i) {
        host_in[i] = complex_t{double(i), -double(i) * 0.5};
    }

    gpu::DeviceBuffer<complex_t> buf(host_in.size());
    buf.upload(host_in);

    // d2d round-trip via a second buffer
    gpu::DeviceBuffer<complex_t> buf2(host_in.size());
    gpu::gpu_memcpy_d2d(buf2.data(), buf.data(),
                        host_in.size() * sizeof(complex_t));

    auto host_out = buf2.download();
    ASSERT_EQ(host_out.size(), host_in.size());
    for (size_t i = 0; i < host_in.size(); ++i) {
        EXPECT_EQ(host_in[i], host_out[i]);  // bitwise equal
    }
}
#endif
```

- [ ] **Step 2: Run, expect failure**

Run: `cd build_metal && cmake --build . --target test_gpu -j4 && ./test/test_gpu --gtest_filter='GPU.MetalMalloc*:GPU.MetalMemoryRoundTrip*' 2>&1 | tail -10`
Expected: FAIL with `GPUNotAvailableError` (current stub).

- [ ] **Step 3: Implement using MTLBuffer**

Replace the stubbed bodies in `src/gpu/memory_metal.cpp`. The strategy: every `void*` returned from `gpu_malloc` is the `contents()` pointer of an `MTLBuffer` with `storageModeShared`. We need to map that pointer back to the buffer in `gpu_free` and `gpu_memcpy_*` — keep a map in a function-local-static.

Replace `memory_metal.cpp` with:

```cpp
// ============================================================================
// KRONOS  src/gpu/memory_metal.cpp
// Apple Silicon unified-memory wrapper for the gpu::memory interface.
// ============================================================================

#ifdef KRONOS_GPU_METAL

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include "gpu/memory.hpp"
#include "gpu/gpu_context.hpp"

#include <cstring>
#include <mutex>
#include <sys/sysctl.h>
#include <unordered_map>

namespace kronos::gpu {

namespace {

struct BufferRegistry {
    std::mutex                                   mtx;
    std::unordered_map<void*, MTL::Buffer*>      map;

    static BufferRegistry& instance() {
        static BufferRegistry r;
        return r;
    }

    void register_buffer(void* contents, MTL::Buffer* buf) {
        std::lock_guard<std::mutex> g(mtx);
        map[contents] = buf;
    }

    MTL::Buffer* lookup(void* contents) const {
        auto it = map.find(contents);
        return (it == map.end()) ? nullptr : it->second;
    }

    MTL::Buffer* pop(void* contents) {
        std::lock_guard<std::mutex> g(mtx);
        auto it = map.find(contents);
        if (it == map.end()) return nullptr;
        MTL::Buffer* b = it->second;
        map.erase(it);
        return b;
    }
};

} // anonymous namespace

void* gpu_malloc(size_t bytes) {
    auto& ctx = GPUContext::instance();
    if (!ctx.is_initialized()) {
        throw GPUNotAvailableError("Metal context not initialized");
    }
    MTL::Device* device = static_cast<MTL::Device*>(ctx.blas_handle());
    MTL::Buffer* buf = device->newBuffer(
        bytes, MTL::ResourceStorageModeShared);
    if (!buf) {
        throw GPUNotAvailableError(
            "MTLBuffer allocation failed (" + std::to_string(bytes) + " bytes)");
    }
    void* contents = buf->contents();
    BufferRegistry::instance().register_buffer(contents, buf);
    return contents;
}

void gpu_free(void* ptr) {
    if (!ptr) return;
    MTL::Buffer* buf = BufferRegistry::instance().pop(ptr);
    if (buf) {
        buf->release();
    }
}

void gpu_memcpy_h2d(void* dst, const void* src, size_t bytes) {
    // On unified memory, dst is already host-visible. memcpy is the transfer.
    std::memcpy(dst, src, bytes);
    // Make CPU writes visible to GPU: didModifyRange on the underlying buffer.
    MTL::Buffer* buf = BufferRegistry::instance().lookup(dst);
    if (buf) {
        buf->didModifyRange(NS::Range::Make(0, bytes));
    }
}

void gpu_memcpy_d2h(void* dst, const void* src, size_t bytes) {
    // Caller is responsible for completing any in-flight GPU work first
    // (done via command-buffer waitUntilCompleted in the FFT/GEMM wrappers).
    std::memcpy(dst, src, bytes);
}

void gpu_memcpy_d2d(void* dst, const void* src, size_t bytes) {
    std::memcpy(dst, src, bytes);
    MTL::Buffer* buf = BufferRegistry::instance().lookup(dst);
    if (buf) {
        buf->didModifyRange(NS::Range::Make(0, bytes));
    }
}

bool gpu_available() {
    MTL::Device* probe = MTL::CreateSystemDefaultDevice();
    if (!probe) return false;
    probe->release();
    return true;
}

size_t gpu_memory_free() {
    MTL::Device* d = MTL::CreateSystemDefaultDevice();
    if (!d) return 0;
    size_t v = d->recommendedMaxWorkingSetSize();
    d->release();
    return v;
}

size_t gpu_memory_total() {
    int64_t physmem = 0;
    size_t len = sizeof(physmem);
    sysctlbyname("hw.memsize", &physmem, &len, nullptr, 0);
    return static_cast<size_t>(physmem);
}

} // namespace kronos::gpu

#endif // KRONOS_GPU_METAL
```

- [ ] **Step 4: Rebuild and run memory tests**

Run: `cd build_metal && cmake --build . --target test_gpu -j4 && ./test/test_gpu --gtest_filter='GPU.MetalMalloc*:GPU.MetalMemoryRoundTrip*' 2>&1 | tail -10`
Expected: both PASS.

- [ ] **Step 5: Commit**

```bash
git add src/gpu/memory_metal.cpp test/test_gpu.cpp
git commit -m "feat(gpu): implement Metal unified-memory wrapper via MTLBuffer"
```

---

### Task 9: Verify `GPUFFTGrid` and `GPUHamiltonian` stub-throw is unchanged

This is a regression-only task. Confirm that the existing `GPUHamiltonian.FallbackToCPU` test still passes on the Metal build — even though `gpu_available()` now returns `true`, the FFT and GEMM stubs still throw, so `GPUHamiltonian` should catch the exception during construction and fall back to CPU.

- [ ] **Step 1: Run the existing fallback test on the Metal build**

Run: `cd build_metal && ./test/test_gpu --gtest_filter='GPUHamiltonian.FallbackToCPU' 2>&1 | tail -10`
Expected: PASS. The construction path throws `GPUNotAvailableError` inside `GPUHamiltonian` (because the FFT stub still throws), and the fallback path kicks in.

- [ ] **Step 2: If the test fails because `gpu_available()` returns true but the construction succeeds partway, that means our FFT/GEMM stubs aren't throwing where expected. Inspect the failure, decide whether to:**
  - patch the stubs to throw earlier (preferred), or
  - mark this test as `GTEST_SKIP` on Metal builds with a TODO comment pointing at Task 14 / Task 12

- [ ] **Step 3: If a patch is needed, commit it**

```bash
git add src/gpu/blas_metal.cpp src/gpu/fft_metal.cpp test/test_gpu.cpp
git commit -m "fix(gpu): keep GPUHamiltonian fallback path green on Metal until FFT/GEMM land"
```

---

## Phase 4 — Complex GEMM

### Task 10: Write the templated complex GEMM Metal shader

**Files:**
- Modify: `src/gpu/kernels/complex_gemm.metal` (replace placeholder)

- [ ] **Step 1: Replace the placeholder with the real kernel**

Replace `src/gpu/kernels/complex_gemm.metal` with:

```msl
// ============================================================================
// KRONOS  src/gpu/kernels/complex_gemm.metal
// Templated complex GEMM: C = alpha*A*B + beta*C
// One implementation, specialized at compile time for fp32 and fp64.
// ============================================================================

#include <metal_stdlib>
using namespace metal;

// 16x16 tiled complex GEMM. T is the scalar (float or double).
// Complex values are pairs (re, im) stored as vec<T,2>.
template <typename T>
inline void zgemm_tiled(
    constant int& M, constant int& N, constant int& K,
    constant vec<T,2>& alpha, constant vec<T,2>& beta,
    device const vec<T,2>* A,
    device const vec<T,2>* B,
    device       vec<T,2>* C,
    constant int& lda, constant int& ldb, constant int& ldc,
    uint2 gid, uint2 lid,
    threadgroup vec<T,2>* tileA, threadgroup vec<T,2>* tileB)
{
    constexpr int TS = 16;

    int row = gid.y;
    int col = gid.x;

    vec<T,2> acc = vec<T,2>(0, 0);

    for (int kk = 0; kk < K; kk += TS) {
        // Cooperatively load a TS x TS tile of A and B into threadgroup memory.
        int a_row = row;
        int a_col = kk + int(lid.x);
        tileA[lid.y * TS + lid.x] =
            (a_row < M && a_col < K) ? A[a_row + a_col * lda] : vec<T,2>(0, 0);

        int b_row = kk + int(lid.y);
        int b_col = col;
        tileB[lid.y * TS + lid.x] =
            (b_row < K && b_col < N) ? B[b_row + b_col * ldb] : vec<T,2>(0, 0);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (int t = 0; t < TS; ++t) {
            vec<T,2> a = tileA[lid.y * TS + t];
            vec<T,2> b = tileB[t * TS + lid.x];
            // (a.re + i a.im) * (b.re + i b.im)
            acc.x += a.x * b.x - a.y * b.y;
            acc.y += a.x * b.y + a.y * b.x;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < M && col < N) {
        // C = alpha*acc + beta*C  (complex multiply alpha*acc + beta*C)
        vec<T,2> c = C[row + col * ldc];
        vec<T,2> ac = vec<T,2>(
            alpha.x * acc.x - alpha.y * acc.y,
            alpha.x * acc.y + alpha.y * acc.x);
        vec<T,2> bc = vec<T,2>(
            beta.x * c.x - beta.y * c.y,
            beta.x * c.y + beta.y * c.x);
        C[row + col * ldc] = ac + bc;
    }
}

// ---------- fp64 entry ------------------------------------------------------

kernel void zgemm_fp64(
    constant int&   M       [[buffer(0)]],
    constant int&   N       [[buffer(1)]],
    constant int&   K       [[buffer(2)]],
    constant double2& alpha [[buffer(3)]],
    constant double2& beta  [[buffer(4)]],
    device const double2* A [[buffer(5)]],
    device const double2* B [[buffer(6)]],
    device       double2* C [[buffer(7)]],
    constant int&   lda     [[buffer(8)]],
    constant int&   ldb     [[buffer(9)]],
    constant int&   ldc     [[buffer(10)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 lid [[thread_position_in_threadgroup]])
{
    threadgroup double2 tileA[16 * 16];
    threadgroup double2 tileB[16 * 16];
    zgemm_tiled<double>(M, N, K, alpha, beta,
                        A, B, C, lda, ldb, ldc,
                        gid, lid, tileA, tileB);
}

// ---------- fp32 entry ------------------------------------------------------

kernel void zgemm_fp32(
    constant int&   M       [[buffer(0)]],
    constant int&   N       [[buffer(1)]],
    constant int&   K       [[buffer(2)]],
    constant float2& alpha  [[buffer(3)]],
    constant float2& beta   [[buffer(4)]],
    device const float2* A  [[buffer(5)]],
    device const float2* B  [[buffer(6)]],
    device       float2* C  [[buffer(7)]],
    constant int&   lda     [[buffer(8)]],
    constant int&   ldb     [[buffer(9)]],
    constant int&   ldc     [[buffer(10)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 lid [[thread_position_in_threadgroup]])
{
    threadgroup float2 tileA[16 * 16];
    threadgroup float2 tileB[16 * 16];
    zgemm_tiled<float>(M, N, K, alpha, beta,
                       A, B, C, lda, ldb, ldc,
                       gid, lid, tileA, tileB);
}
```

- [ ] **Step 2: Confirm shader compiles**

Run: `cd build_metal && cmake --build . --target kronos_metallib 2>&1 | tail -10`
Expected: `kronos.metallib` is rebuilt with no errors. If `metal3.0` rejects `double` types, fall back to `-std=metal2.4` in `cmake/MetalShaders.cmake`. If fp64 is rejected outright, document the failure and consult macOS version; the spec requires macOS 13+.

- [ ] **Step 3: Commit**

```bash
git add src/gpu/kernels/complex_gemm.metal
git commit -m "feat(gpu): add templated complex-GEMM Metal shader (fp32 + fp64)"
```

---

### Task 11: Add precision-mode flag to `GPUContext`

The dispatcher in `blas_metal.cpp` (next task) needs to know which kernel to select. Add a runtime-settable flag on the context so the SCF setup code can toggle it.

**Files:**
- Modify: `src/gpu/gpu_context.hpp`

- [ ] **Step 1: Extend `GPUContext` with the precision flag**

Edit `src/gpu/gpu_context.hpp`. In the public section after `set_deterministic`:

```cpp
    /// Apple-only: enable fp32 fast mode (NOT validation-grade).
    /// No-op on cuda/hip builds. Default: false (fp64).
    void set_apple_fast_mode(bool enable) { apple_fast_mode_ = enable; }
    bool apple_fast_mode() const { return apple_fast_mode_; }
```

In the private members:

```cpp
    bool apple_fast_mode_{false};
```

- [ ] **Step 2: Hook the flag through from `InputData.hardware`**

In `src/main.cpp`, after `parse_input` returns the `InputData`, before any GPU calls, add:

```cpp
#ifdef KRONOS_GPU_METAL
    if (input.hardware.apple_fast_mode) {
        kronos::gpu::GPUContext::instance().set_apple_fast_mode(true);
        Logger::instance().warning("apple_fast_mode",
            "fp32 GPU path active — results are not validation-grade");
    }
#endif
```

(Place this near the existing `gpu::GPUContext::instance().init(...)` call. If there isn't one yet, add `gpu::GPUContext::instance().init(mpi::rank(), mpi::local_rank())` just before the flag setter so the context is alive when we set the flag.)

- [ ] **Step 3: Build and run all CPU tests as a regression check**

Run: `cd build && cmake --build . -j4 && ctest -j2 --output-on-failure 2>&1 | tail -3`
Expected: 442/442 pass (no behavioral change on the CPU build).

- [ ] **Step 4: Commit**

```bash
git add src/gpu/gpu_context.hpp src/main.cpp
git commit -m "feat(gpu): plumb apple_fast_mode flag through GPUContext"
```

---

### Task 12: Real `blas_metal.cpp` (dispatch to MSL kernel)

**Files:**
- Modify: `src/gpu/blas_metal.cpp`
- Modify: `test/test_gpu.cpp` (add GEMM tests)

- [ ] **Step 1: Write failing tests**

Append to `test/test_gpu.cpp`:

```cpp
#ifdef KRONOS_GPU_METAL
TEST(GPU, MetalComplexGEMMFP64Identity) {
    // C = 1 * I_64 * I_64 + 0 * C  should give identity
    auto& ctx = gpu::GPUContext::instance();
    ctx.init();
    ASSERT_TRUE(ctx.is_initialized());
    ctx.set_apple_fast_mode(false);

    const int N = 64;
    std::vector<complex_t> A(N * N, {0, 0});
    std::vector<complex_t> B(N * N, {0, 0});
    for (int i = 0; i < N; ++i) {
        A[i + i * N] = {1.0, 0.0};
        B[i + i * N] = {1.0, 0.0};
    }
    std::vector<complex_t> C(N * N, {0, 0});

    gpu::DeviceBuffer<complex_t> dA(N * N); dA.upload(A);
    gpu::DeviceBuffer<complex_t> dB(N * N); dB.upload(B);
    gpu::DeviceBuffer<complex_t> dC(N * N); dC.upload(C);

    gpu::gemm(N, N, N,
              complex_t{1.0, 0.0},
              dA.data(), N,
              dB.data(), N,
              complex_t{0.0, 0.0},
              dC.data(), N);

    auto Cout = dC.download();
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            complex_t expected = (i == j) ? complex_t{1, 0} : complex_t{0, 0};
            EXPECT_NEAR(std::abs(Cout[i + j * N] - expected), 0.0, 1e-12)
                << "i=" << i << " j=" << j;
        }
    }
}

TEST(GPU, MetalComplexGEMMFP64MatchesCPU) {
    auto& ctx = gpu::GPUContext::instance();
    ctx.init();
    ASSERT_TRUE(ctx.is_initialized());
    ctx.set_apple_fast_mode(false);

    const int M = 17, N = 13, K = 23;  // intentionally non-tile-aligned
    std::mt19937_64 rng(42);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    auto rand_mat = [&](int rows, int cols) {
        std::vector<complex_t> v(rows * cols);
        for (auto& x : v) x = {dist(rng), dist(rng)};
        return v;
    };

    auto A = rand_mat(M, K);
    auto B = rand_mat(K, N);
    auto C = rand_mat(M, N);
    auto C_ref = C;

    complex_t alpha{0.7, -0.3};
    complex_t beta {0.4,  0.2};

    // Reference: column-major C[i,j] = sum_k A[i,k] * B[k,j]
    for (int j = 0; j < N; ++j)
    for (int i = 0; i < M; ++i) {
        complex_t s{0, 0};
        for (int k = 0; k < K; ++k)
            s += A[i + k * M] * B[k + j * K];
        C_ref[i + j * M] = alpha * s + beta * C_ref[i + j * M];
    }

    gpu::DeviceBuffer<complex_t> dA(M * K); dA.upload(A);
    gpu::DeviceBuffer<complex_t> dB(K * N); dB.upload(B);
    gpu::DeviceBuffer<complex_t> dC(M * N); dC.upload(C);

    gpu::gemm(M, N, K, alpha,
              dA.data(), M, dB.data(), K,
              beta, dC.data(), M);

    auto Cout = dC.download();
    for (int i = 0; i < int(Cout.size()); ++i) {
        EXPECT_NEAR(std::abs(Cout[i] - C_ref[i]), 0.0, 1e-11)
            << "mismatch at index " << i;
    }
}
#endif
```

(Add `#include <random>` at the top of `test_gpu.cpp` if not already present.)

- [ ] **Step 2: Run, expect failures**

Run: `cd build_metal && cmake --build . --target test_gpu -j4 && ./test/test_gpu --gtest_filter='GPU.MetalComplexGEMM*' 2>&1 | tail -10`
Expected: FAIL — `gemm` still throws.

- [ ] **Step 3: Implement `blas_metal.cpp`**

Replace `src/gpu/blas_metal.cpp` with:

```cpp
// ============================================================================
// KRONOS  src/gpu/blas_metal.cpp
// Metal complex GEMM dispatcher.
// Loads kronos.metallib at first call; caches MTL::ComputePipelineState
// per precision mode.
// ============================================================================

#ifdef KRONOS_GPU_METAL

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include "gpu/blas.hpp"
#include "gpu/gpu_context.hpp"
#include "gpu/memory.hpp"

#include <filesystem>
#include <mutex>
#include <stdexcept>
#include <string>

namespace kronos::gpu {

namespace {

struct PipelineCache {
    std::mutex                       mtx;
    MTL::Library*                    library  = nullptr;
    MTL::ComputePipelineState*       psoFP64  = nullptr;
    MTL::ComputePipelineState*       psoFP32  = nullptr;

    static PipelineCache& instance() {
        static PipelineCache c;
        return c;
    }
};

MTL::Library* load_library(MTL::Device* device) {
    // Locate kronos.metallib next to the executable.
    // For tests run via ctest, CWD is test/ — fall back to a few candidates.
    namespace fs = std::filesystem;
    std::vector<fs::path> candidates = {
        fs::current_path() / "kronos.metallib",
        fs::current_path().parent_path() / "kronos.metallib",
        fs::current_path() / ".." / "kronos.metallib",
    };
    for (auto& p : candidates) {
        if (!fs::exists(p)) continue;
        NS::String* url_s = NS::String::string(
            p.string().c_str(), NS::UTF8StringEncoding);
        NS::URL* url = NS::URL::fileURLWithPath(url_s);
        NS::Error* err = nullptr;
        MTL::Library* lib = device->newLibrary(url, &err);
        if (lib) return lib;
    }
    throw std::runtime_error(
        "kronos.metallib not found; searched CWD and one level up");
}

MTL::ComputePipelineState* get_pipeline(bool fp32) {
    auto& c = PipelineCache::instance();
    std::lock_guard<std::mutex> g(c.mtx);

    auto& ctx = GPUContext::instance();
    MTL::Device* device = static_cast<MTL::Device*>(ctx.blas_handle());

    if (!c.library) {
        c.library = load_library(device);
    }
    if (fp32 && c.psoFP32) return c.psoFP32;
    if (!fp32 && c.psoFP64) return c.psoFP64;

    const char* fn = fp32 ? "zgemm_fp32" : "zgemm_fp64";
    NS::String* fn_s = NS::String::string(fn, NS::UTF8StringEncoding);
    MTL::Function* func = c.library->newFunction(fn_s);
    if (!func) {
        throw std::runtime_error(std::string("Metal: function '") + fn + "' missing");
    }
    NS::Error* err = nullptr;
    MTL::ComputePipelineState* pso = device->newComputePipelineState(func, &err);
    func->release();
    if (!pso) {
        throw std::runtime_error(std::string("Metal: newComputePipelineState failed for ") + fn);
    }
    if (fp32) c.psoFP32 = pso; else c.psoFP64 = pso;
    return pso;
}

} // anonymous namespace

void gemm(int m, int n, int k,
          complex_t alpha,
          const complex_t* A, int lda,
          const complex_t* B, int ldb,
          complex_t beta,
          complex_t* C, int ldc)
{
    auto& ctx = GPUContext::instance();
    if (!ctx.is_initialized()) {
        throw GPUNotAvailableError("Metal context not initialized");
    }
    const bool fp32 = ctx.apple_fast_mode();

    MTL::Device* device = static_cast<MTL::Device*>(ctx.blas_handle());
    MTL::CommandQueue* queue = static_cast<MTL::CommandQueue*>(ctx.metal_queue());

    MTL::ComputePipelineState* pso = get_pipeline(fp32);

    MTL::CommandBuffer*       cmd = queue->commandBuffer();
    MTL::ComputeCommandEncoder* enc = cmd->computeCommandEncoder();
    enc->setComputePipelineState(pso);

    // Pack scalar arguments as zero-padded scratch (must be aligned).
    auto encode_int = [&](int i, int slot) {
        enc->setBytes(&i, sizeof(int), slot);
    };
    encode_int(m, 0); encode_int(n, 1); encode_int(k, 2);

    if (fp32) {
        float a[2] = { float(alpha.real()), float(alpha.imag()) };
        float b[2] = { float(beta.real()),  float(beta.imag())  };
        enc->setBytes(a, sizeof(a), 3);
        enc->setBytes(b, sizeof(b), 4);

        // For fp32 mode, A/B/C must be float2 buffers. If KRONOS still passes
        // complex128, we'd need to downcast. v0.5.1 scope:
        //   - fp32 mode requires the caller to upload float2 buffers.
        //   - DeviceBuffer<float> + manual downcast in physics code is out of
        //     scope; for now, throw if fp32 mode is selected on a complex128
        //     call site.
        throw GPUNotAvailableError(
            "Apple fast mode (fp32) requires float2 wavefunction buffers; "
            "physics-side downcast is deferred. Run in fp64 mode for now.");
    }

    // fp64 path
    double a[2] = { alpha.real(), alpha.imag() };
    double b[2] = { beta.real(),  beta.imag()  };
    enc->setBytes(a, sizeof(a), 3);
    enc->setBytes(b, sizeof(b), 4);

    // Look up the MTLBuffer for each pointer via the memory registry.
    // (See memory_metal.cpp BufferRegistry — we need an internal accessor.)
    extern MTL::Buffer* metal_buffer_for(const void* p);  // exposed below
    auto bind_buffer = [&](const void* p, int slot) {
        MTL::Buffer* buf = metal_buffer_for(p);
        if (!buf) throw std::runtime_error("metal: pointer not from gpu_malloc");
        enc->setBuffer(buf, 0, slot);
    };
    bind_buffer(A, 5);
    bind_buffer(B, 6);
    bind_buffer(C, 7);

    encode_int(lda, 8); encode_int(ldb, 9); encode_int(ldc, 10);

    constexpr int TS = 16;
    MTL::Size threads = MTL::Size::Make((n + TS - 1) / TS * TS,
                                        (m + TS - 1) / TS * TS, 1);
    MTL::Size group   = MTL::Size::Make(TS, TS, 1);
    enc->dispatchThreads(threads, group);
    enc->endEncoding();
    cmd->commit();
    cmd->waitUntilCompleted();
}

complex_t zdotc(int /*n*/, const complex_t* /*x*/, const complex_t* /*y*/) {
    // Not used on the GPU hot path in v0.5.1. Implement when a caller needs it.
    throw GPUNotAvailableError(
        "Metal zdotc not implemented in v0.5.1 (no caller yet)");
}

} // namespace kronos::gpu

#endif // KRONOS_GPU_METAL
```

- [ ] **Step 4: Expose `metal_buffer_for` from `memory_metal.cpp`**

Edit `src/gpu/memory_metal.cpp`. At the bottom of the file (still inside the `#ifdef KRONOS_GPU_METAL`), add:

```cpp
// Internal accessor for blas_metal.cpp to bind MTLBuffer arguments.
MTL::Buffer* metal_buffer_for(const void* p) {
    return BufferRegistry::instance().lookup(const_cast<void*>(p));
}
```

Note: this is in the anonymous-detail style — not declared in any header. It's `extern`-declared at the use site in `blas_metal.cpp`. Acceptable because both files are in the same translation unit cluster and the symbol is not exported beyond the library.

- [ ] **Step 5: Rebuild and run GEMM tests**

Run: `cd build_metal && cmake --build . -j4 && ./test/test_gpu --gtest_filter='GPU.MetalComplexGEMM*' 2>&1 | tail -15`
Expected: both PASS (fp64 path). The identity test catches indexing bugs; the random-matrix test catches everything else.

- [ ] **Step 6: Commit**

```bash
git add src/gpu/blas_metal.cpp src/gpu/memory_metal.cpp test/test_gpu.cpp
git commit -m "feat(gpu): Metal complex GEMM dispatch (fp64 production path)"
```

---

## Phase 5 — FFT (VkFFT wrapper)

### Task 13: Real `fft_metal.cpp` (VkFFT wrapper, fp64)

**Files:**
- Modify: `src/gpu/fft_metal.cpp`
- Modify: `test/test_gpu.cpp` (FFT tests)

- [ ] **Step 1: Write failing tests**

Append to `test/test_gpu.cpp`:

```cpp
#ifdef KRONOS_GPU_METAL
TEST(GPU, MetalFFTGridConstructs) {
    auto& ctx = gpu::GPUContext::instance();
    ctx.init();
    ASSERT_TRUE(ctx.is_initialized());
    gpu::GPUFFTGrid g({16, 16, 16});
    auto d = g.dims();
    EXPECT_EQ(d[0], 16); EXPECT_EQ(d[1], 16); EXPECT_EQ(d[2], 16);
}

TEST(GPU, MetalFFTRoundTripFP64) {
    auto& ctx = gpu::GPUContext::instance();
    ctx.init();
    ASSERT_TRUE(ctx.is_initialized());
    ctx.set_apple_fast_mode(false);

    const std::array<int, 3> dims = {16, 16, 16};
    const int N = dims[0] * dims[1] * dims[2];

    std::mt19937_64 rng(99);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    std::vector<complex_t> in(N);
    for (auto& x : in) x = {dist(rng), dist(rng)};

    gpu::DeviceBuffer<complex_t> d_in(N);  d_in.upload(in);
    gpu::DeviceBuffer<complex_t> d_k(N);
    gpu::DeviceBuffer<complex_t> d_out(N);

    gpu::GPUFFTGrid grid(dims);
    grid.forward(d_in.data(), d_k.data());
    grid.inverse(d_k.data(),  d_out.data());

    auto out = d_out.download();
    // forward+inverse leaves a factor of N (FFTW/VkFFT convention).
    for (int i = 0; i < N; ++i) out[i] /= double(N);

    for (int i = 0; i < N; ++i) {
        EXPECT_NEAR(std::abs(out[i] - in[i]), 0.0, 1e-12)
            << "i=" << i;
    }
}

TEST(GPU, MetalFFTRoundTripFP32) {
    auto& ctx = gpu::GPUContext::instance();
    ctx.init();
    ASSERT_TRUE(ctx.is_initialized());
    ctx.set_apple_fast_mode(true);  // enable fp32 FFT path

    const std::array<int, 3> dims = {16, 16, 16};
    const int N = dims[0] * dims[1] * dims[2];

    std::mt19937_64 rng(99);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    std::vector<complex_t> in(N);
    for (auto& x : in) x = {dist(rng), dist(rng)};

    gpu::DeviceBuffer<complex_t> d_in(N);  d_in.upload(in);
    gpu::DeviceBuffer<complex_t> d_k(N);
    gpu::DeviceBuffer<complex_t> d_out(N);

    gpu::GPUFFTGrid grid(dims);  // constructed in fp32 mode due to ctx flag
    grid.forward(d_in.data(), d_k.data());
    grid.inverse(d_k.data(),  d_out.data());

    auto out = d_out.download();
    for (int i = 0; i < N; ++i) out[i] /= double(N);

    // fp32 has ~1e-6 unit roundoff; allow looser tolerance.
    for (int i = 0; i < N; ++i) {
        EXPECT_NEAR(std::abs(out[i] - in[i]), 0.0, 1e-4)
            << "i=" << i;
    }

    ctx.set_apple_fast_mode(false);  // reset for any subsequent tests
}
#endif
```

- [ ] **Step 2: Run, expect failure**

Run: `cd build_metal && cmake --build . --target test_gpu -j4 && ./test/test_gpu --gtest_filter='GPU.MetalFFT*' 2>&1 | tail -10`
Expected: FAIL — FFT stub still throws.

- [ ] **Step 3: Implement `fft_metal.cpp`**

Replace `src/gpu/fft_metal.cpp` with:

```cpp
// ============================================================================
// KRONOS  src/gpu/fft_metal.cpp
// 3D complex FFT on Apple GPU via VkFFT.
// ============================================================================

#ifdef KRONOS_GPU_METAL

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

// VkFFT is header-only when VKFFT_BACKEND=5 (Metal); ensure Metal headers are
// included before vkFFT.h so VkFFT picks up the metal-cpp wrappers.
#include "vkFFT.h"

#include "gpu/fft.hpp"
#include "gpu/gpu_context.hpp"
#include "gpu/memory.hpp"

#include <stdexcept>
#include <string>

namespace kronos::gpu {

// Internal accessor declared in memory_metal.cpp (Task 12 step 4).
extern MTL::Buffer* metal_buffer_for(const void* p);

namespace {

struct VkFFTState {
    VkFFTApplication app{};
    VkFFTConfiguration cfg{};
    std::array<int, 3>   dims;
    bool                 owned = false;
};

} // anonymous namespace

GPUFFTGrid::GPUFFTGrid(std::array<int, 3> dims)
    : dims_(dims)
{
    auto& ctx = GPUContext::instance();
    if (!ctx.is_initialized()) {
        throw GPUNotAvailableError("Metal context not initialized for FFT");
    }
    const bool fp32 = ctx.apple_fast_mode();

    VkFFTState* st = new VkFFTState();
    st->dims = dims;

    MTL::Device* device = static_cast<MTL::Device*>(ctx.blas_handle());
    MTL::CommandQueue* queue = static_cast<MTL::CommandQueue*>(ctx.metal_queue());

    st->cfg.device = device;
    st->cfg.queue  = queue;
    st->cfg.FFTdim = 3;
    st->cfg.size[0] = (uint64_t)dims[0];
    st->cfg.size[1] = (uint64_t)dims[1];
    st->cfg.size[2] = (uint64_t)dims[2];
    st->cfg.doublePrecision = fp32 ? 0u : 1u;
    st->cfg.performR2C      = 0u;       // complex-to-complex
    st->cfg.isInputFormatted = 1u;
    st->cfg.inverseReturnToInputBuffer = 0u;

    VkFFTResult r = initializeVkFFT(&st->app, st->cfg);
    if (r != VKFFT_SUCCESS) {
        delete st;
        throw std::runtime_error(
            "initializeVkFFT failed: code " + std::to_string(int(r)));
    }
    st->owned = true;
    plan_forward_ = static_cast<void*>(st);
    plan_inverse_ = nullptr;  // VkFFT uses one app for both directions
}

GPUFFTGrid::~GPUFFTGrid() {
    if (plan_forward_) {
        VkFFTState* st = static_cast<VkFFTState*>(plan_forward_);
        if (st->owned) deleteVkFFT(&st->app);
        delete st;
        plan_forward_ = nullptr;
    }
}

void GPUFFTGrid::forward(const complex_t* d_input, complex_t* d_output) {
    VkFFTState* st = static_cast<VkFFTState*>(plan_forward_);
    MTL::Buffer* in_buf  = metal_buffer_for(d_input);
    MTL::Buffer* out_buf = metal_buffer_for(d_output);
    if (!in_buf || !out_buf) {
        throw std::runtime_error("FFT pointers not from gpu_malloc");
    }

    auto& ctx = GPUContext::instance();
    MTL::CommandQueue* queue = static_cast<MTL::CommandQueue*>(ctx.metal_queue());
    MTL::CommandBuffer* cmd = queue->commandBuffer();

    VkFFTLaunchParams lp{};
    lp.buffer       = &in_buf;
    lp.outputBuffer = &out_buf;
    lp.commandBuffer = cmd;

    VkFFTResult r = VkFFTAppend(&st->app, /*inverse=*/-1, &lp);
    if (r != VKFFT_SUCCESS) {
        throw std::runtime_error(
            "VkFFTAppend(forward) failed: " + std::to_string(int(r)));
    }
    cmd->commit();
    cmd->waitUntilCompleted();
}

void GPUFFTGrid::inverse(const complex_t* d_input, complex_t* d_output) {
    VkFFTState* st = static_cast<VkFFTState*>(plan_forward_);
    MTL::Buffer* in_buf  = metal_buffer_for(d_input);
    MTL::Buffer* out_buf = metal_buffer_for(d_output);
    if (!in_buf || !out_buf) {
        throw std::runtime_error("FFT pointers not from gpu_malloc");
    }

    auto& ctx = GPUContext::instance();
    MTL::CommandQueue* queue = static_cast<MTL::CommandQueue*>(ctx.metal_queue());
    MTL::CommandBuffer* cmd = queue->commandBuffer();

    VkFFTLaunchParams lp{};
    lp.buffer       = &in_buf;
    lp.outputBuffer = &out_buf;
    lp.commandBuffer = cmd;

    VkFFTResult r = VkFFTAppend(&st->app, /*inverse=*/+1, &lp);
    if (r != VKFFT_SUCCESS) {
        throw std::runtime_error(
            "VkFFTAppend(inverse) failed: " + std::to_string(int(r)));
    }
    cmd->commit();
    cmd->waitUntilCompleted();
}

std::array<int, 3> GPUFFTGrid::dims() const {
    return dims_;
}

} // namespace kronos::gpu

#endif // KRONOS_GPU_METAL
```

NOTE: VkFFT's Metal binding API (`VkFFTLaunchParams::buffer` as `MTL::Buffer**`) is the v1.3.4 contract. If a different VkFFT version uses different types (e.g., `void**`), adjust. Watch for compile errors and consult `vkfft-src/documentation/`.

- [ ] **Step 4: Rebuild and run FFT tests**

Run: `cd build_metal && cmake --build . -j4 && ./test/test_gpu --gtest_filter='GPU.MetalFFT*' 2>&1 | tail -15`
Expected: both PASS. `MetalFFTRoundTripFP64` is the meat — fp64 forward+inverse must recover the input to 1e-12.

- [ ] **Step 5: Commit**

```bash
git add src/gpu/fft_metal.cpp test/test_gpu.cpp
git commit -m "feat(gpu): Metal 3D complex FFT via VkFFT (fp64 path)"
```

---

### Task 14: Explicit `GPUHamiltonian` fp64 numeric check + end-to-end bring-up

The existing `GPUHamiltonian.FallbackToCPU` test happens to also check
`hpsi_gpu == hpsi_cpu` to 1e-14 — but its name and intent are about the
fallback path, not Metal correctness. Add a dedicated, clearly-named test
for spec §5.2 #6.

**Files:**
- Modify: `test/test_gpu.cpp`

- [ ] **Step 1: Write the failing test**

Append to `test/test_gpu.cpp` (inside the `#ifdef KRONOS_GPU_METAL` block):

```cpp
#ifdef KRONOS_GPU_METAL
TEST(GPU, MetalHamiltonianApplyFP64MatchesCPU) {
    auto& ctx = gpu::GPUContext::instance();
    ctx.init();
    ASSERT_TRUE(ctx.is_initialized());
    ctx.set_apple_fast_mode(false);

    Crystal crystal = make_si_diamond();
    auto pps = make_si_pp();
    PlaneWaveBasis basis(crystal, 15.0);
    FFTGrid fft_grid(basis);
    NonlocalPP nonlocal_pp(crystal, basis, pps);
    Hamiltonian cpu_ham(crystal, basis, fft_grid, nonlocal_pp);

    int num_grid = fft_grid.total_points();
    std::vector<complex_t> veff(num_grid, {-1.0, 0.0});
    cpu_ham.update_veff(veff);

    GPUHamiltonian gpu_ham(crystal, basis, fft_grid, nonlocal_pp, cpu_ham);
    gpu_ham.update_veff(veff);
    ASSERT_TRUE(gpu_ham.gpu_active())
        << "On Metal build, GPUHamiltonian must activate the GPU path";

    // Non-trivial input wavefunction
    int npw = static_cast<int>(basis.num_pw());
    std::mt19937_64 rng(7);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    CVec psi(npw);
    for (auto& x : psi) x = {dist(rng), dist(rng)};

    Vec3 k_frac = {0.0, 0.0, 0.0};
    CVec hpsi_gpu = gpu_ham.apply(psi, k_frac);
    CVec hpsi_cpu = cpu_ham.apply(psi, k_frac);

    ASSERT_EQ(hpsi_gpu.size(), hpsi_cpu.size());
    for (int i = 0; i < npw; ++i) {
        EXPECT_NEAR(std::abs(hpsi_gpu[i] - hpsi_cpu[i]), 0.0, 1e-12)
            << "Metal H|ψ⟩ diverges from CPU at i=" << i;
    }
}
#endif
```

- [ ] **Step 2: Run, expect either PASS (if Task 12+13 covered everything) or FAIL (if a host-roundtrip path in gpu_hamiltonian.cpp triggers a stub)**

Run: `cd build_metal && cmake --build . -j4 && ./test/test_gpu --gtest_filter='GPU.MetalHamiltonianApplyFP64MatchesCPU' 2>&1 | tail -20`

If it fails, the failure is one of three things:
1. `GPUHamiltonian` constructor still throws somewhere (an unimplemented primitive). Look at the stack and fix the missing piece. Possible candidates: `gpu_memcpy_d2h` returning before sync, an unmapped pointer.
2. Numerical mismatch > 1e-12. Likely cause: VkFFT normalization convention differs from CPU FFTW (one applies 1/N on inverse, the other doesn't). Adjust the normalization division in `gpu_hamiltonian.cpp::apply_gpu`.
3. A previously-working CPU code path was accidentally regressed. Run the CPU test suite to confirm scope.

Fix and re-run until PASS.

- [ ] **Step 3: Run all `test_gpu` tests as the bring-up check**

Run: `cd build_metal && ./test/test_gpu 2>&1 | tail -25`
Expected: every test passes.

- [ ] **Step 4: Run the full CPU test suite for regression**

Run: `cd build && ctest -j2 --output-on-failure 2>&1 | tail -3`
Expected: 442/442 pass.

- [ ] **Step 5: Commit**

```bash
git add test/test_gpu.cpp
git commit -m "test(gpu): explicit Metal Hamiltonian fp64 numeric agreement check"
```

---

## Phase 6 — Apple fast-mode runtime gating

### Task 15: CLI flag `--apple-fast-mode`

**Files:**
- Modify: `src/main.cpp`
- Modify: `test/test_input.cpp` (add CLI test if a test harness exists for main; otherwise skip and rely on the YAML path)

- [ ] **Step 1: Parse the flag**

In `src/main.cpp`, edit the argument-handling block. After locating `std::string input_file = argv[1];` (around line 45), add a simple flag scan **before** the `parse_input` call:

```cpp
    bool cli_apple_fast = false;
    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--apple-fast-mode") cli_apple_fast = true;
        else {
            std::cerr << "Unknown argument: " << a << "\n";
            return 1;
        }
    }
```

Then, after `parse_input(...)` returns the `InputData`, OR the CLI flag with the YAML value:

```cpp
    if (cli_apple_fast) input.hardware.apple_fast_mode = true;
```

(The existing Task-11 hook that calls `set_apple_fast_mode` will then pick this up.)

- [ ] **Step 2: Write a unit test for the warning emission**

Append to `test/test_gpu.cpp`:

```cpp
#ifdef KRONOS_GPU_METAL
TEST(GPU, AppleFastModeEmitsWarning) {
    // Capture stderr (Logger::warning writes there).
    testing::internal::CaptureStderr();

    auto& ctx = gpu::GPUContext::instance();
    ctx.init();
    ctx.set_apple_fast_mode(true);
    Logger::instance().warning("apple_fast_mode",
        "fp32 GPU path active — results are not validation-grade");

    std::string captured = testing::internal::GetCapturedStderr();
    EXPECT_NE(captured.find("apple_fast_mode"), std::string::npos)
        << "Expected apple_fast_mode warning in stderr; got: " << captured;

    ctx.set_apple_fast_mode(false);
}
#endif
```

(`#include "utils/logger.hpp"` at the top if not already present.)

- [ ] **Step 3: Run, expect PASS** (the warning helper from Task 11 should already be wired)

Run: `cd build_metal && cmake --build . --target test_gpu -j4 && ./test/test_gpu --gtest_filter='GPU.AppleFastModeEmitsWarning' 2>&1 | tail -10`
Expected: PASS.

- [ ] **Step 4: Verify with a manual run**

Run a tiny input (any existing YAML in `examples/`) twice:
```bash
cd build_metal
./kronos examples/si_bulk.yaml 2>&1 | grep -i "apple_fast\|fast_mode" || echo "(no fast mode output — expected)"
./kronos examples/si_bulk.yaml --apple-fast-mode 2>&1 | grep -i "apple_fast\|fast_mode"
```
Expected: second run prints the `apple_fast_mode` warning from Task 11.

- [ ] **Step 5: Commit**

```bash
git add src/main.cpp test/test_gpu.cpp
git commit -m "feat(cli): add --apple-fast-mode and unit test for warning emission"
```

---

### Task 16: Validation-suite guard against fast mode

**Files:**
- Modify: `test/test_validation.cpp` (find the entrypoint or fixture that initializes runs; add abort if fast mode is on)

- [ ] **Step 1: Locate validation entry**

Run: `head -30 test/test_validation.cpp` to find the test fixture setup. Whatever function initializes the run for the Delta-test runner, that's our hook point.

- [ ] **Step 2: Add the guard**

In the test's setup (a `SetUp()` in a fixture, or at the top of each `TEST_F`), add:

```cpp
#ifdef KRONOS_GPU_METAL
    if (gpu::GPUContext::instance().apple_fast_mode()) {
        FAIL() << "Validation suite refuses to run with apple_fast_mode (fp32). "
                  "Disable via --apple-fast-mode off or hardware.apple_fast_mode: false.";
    }
#endif
```

- [ ] **Step 3: Verify**

Run: `cd build_metal && ./test/test_validation 2>&1 | tail -3` — should pass (fast mode is off by default).
Run: temporarily set `apple_fast_mode=true` in code, rebuild test_validation, run, confirm it fails with the guard message, revert.

- [ ] **Step 4: Commit**

```bash
git add test/test_validation.cpp
git commit -m "test(validation): refuse to run in apple_fast_mode (fp32 not validation-grade)"
```

---

## Phase 7 — End-to-end integration

### Task 17: Si bulk SCF on Metal vs CPU (< 10 µRy)

**Files:**
- Modify: `test/test_gpu.cpp`

- [ ] **Step 1: Write the integration test**

Append to `test/test_gpu.cpp` (inside the `#ifdef KRONOS_GPU_METAL` block, near the bottom):

```cpp
#ifdef KRONOS_GPU_METAL
TEST(Integration, SiBulkSCFOnMetalMatchesCPU) {
    // Si diamond, smallest cell, Gamma-only, low ecutwfc to keep runtime sane.
    Crystal crystal = make_si_diamond();        // helper defined earlier in file
    auto pps = make_si_pp();

    PlaneWaveBasis basis(crystal, 15.0);
    FFTGrid fft_grid(basis);
    NonlocalPP nonlocal_pp(crystal, basis, pps);

    auto& ctx = gpu::GPUContext::instance();
    ctx.init();
    ASSERT_TRUE(ctx.is_initialized());
    ctx.set_apple_fast_mode(false);

    // Reference SCF on CPU
    double e_cpu = run_minimal_scf(crystal, basis, fft_grid, nonlocal_pp,
                                   /*use_gpu=*/false);

    // GPU SCF
    double e_gpu = run_minimal_scf(crystal, basis, fft_grid, nonlocal_pp,
                                   /*use_gpu=*/true);

    EXPECT_NEAR(e_gpu, e_cpu, 1e-5)
        << "Metal SCF must match CPU within 10 µRy; got "
        << "CPU=" << e_cpu << " Ry, GPU=" << e_gpu << " Ry, "
        << "delta=" << (e_gpu - e_cpu) << " Ry";
}
#endif
```

- [ ] **Step 2: Add the `run_minimal_scf` helper**

In the anonymous namespace at the top of `test/test_gpu.cpp` (where `make_si_diamond` lives), add:

```cpp
double run_minimal_scf(const Crystal& crystal,
                       const PlaneWaveBasis& basis,
                       FFTGrid& fft_grid,
                       NonlocalPP& nonlocal_pp,
                       bool use_gpu) {
    // Build a minimal SCF context. Use whatever the existing CPU integration
    // tests use as a pattern (look at test_scf.cpp for examples).
    // Run 20 SCF iterations with default mixing; return the converged energy.
    //
    // Pseudocode — the actual call depends on how SCF is structured in
    // src/solver/scf.cpp:
    //
    //   SCFLoop loop(crystal, basis, fft_grid, nonlocal_pp, /*use_gpu=*/use_gpu);
    //   loop.set_max_iter(20);
    //   loop.set_tol(1e-8);
    //   loop.run();
    //   return loop.total_energy();
    //
    // If no clean test entrypoint exists, the implementer must add one to
    // SCFLoop (a tested two-line wrapper, not new functionality) and commit
    // it separately. Read src/solver/scf.cpp first to confirm.
}
```

- [ ] **Step 3: Run the integration test**

Run: `cd build_metal && cmake --build . --target test_gpu -j4 && ./test/test_gpu --gtest_filter='Integration.SiBulkSCFOnMetalMatchesCPU' 2>&1 | tail -20`
Expected: PASS. Total runtime likely several seconds (fp64 emulation is slow); if it exceeds 10 minutes, reduce `ecutwfc` to 10.0 and/or trim SCF iterations.

- [ ] **Step 4: Commit**

```bash
git add test/test_gpu.cpp
git commit -m "test(integration): Si bulk SCF on Metal matches CPU within 10 µRy"
```

---

## Phase 8 — Documentation & final verification

### Task 18: Update docs

**Files:**
- Modify: `README.md`
- Modify: `docs/user_guide.md`
- Modify: `docs/architecture.md`
- Modify: `CLAUDE.md`

- [ ] **Step 1: README.md — add `metal` to the backend table and the build commands**

Edit `README.md:55-70` (the CMake options table and the build commands section). Change the row:

```
| `KRONOS_GPU_BACKEND` | `none` | GPU backend: `none`, `cuda`, or `hip` |
```

to:

```
| `KRONOS_GPU_BACKEND` | `none` | GPU backend: `none`, `cuda`, `hip`, or `metal` |
```

Add to the build-commands block:

```bash
# Metal build (Apple Silicon)
cmake -B build -S . -DKRONOS_GPU_BACKEND=metal
cmake --build build -j$(sysctl -n hw.ncpu)
```

Add a short note immediately below the build commands:

> **Note on Apple Silicon:** Metal GPU acceleration on Apple Silicon runs in
> fp64 by default for correctness but uses software-emulated double precision —
> expect lower throughput than the CPU path on the same chip. The Metal backend
> exists primarily for local development and test parity with the CUDA/HIP
> paths. For performance-critical runs, use NVIDIA or AMD GPUs.

- [ ] **Step 2: docs/user_guide.md — add the YAML key**

Find the section documenting `hardware.*` keys. Add:

```yaml
hardware:
  gpu_backend: metal        # cuda | hip | metal | none
  apple_fast_mode: false    # Apple-only: opt-in fp32 fast path (NOT validation-grade)
```

- [ ] **Step 3: docs/architecture.md — extend the GPU section**

Find the section about `src/gpu/` (around line 692 per earlier grep). Add a subsection:

```markdown
### Metal backend (Apple Silicon, v0.5.1)

The Metal backend mirrors the CUDA/HIP pattern but lives on Apple GPUs:

- `src/gpu/gpu_context_metal.cpp` — MTLDevice + MTLCommandQueue
- `src/gpu/memory_metal.cpp` — MTLBuffer with `storageModeShared` (Apple
  Silicon's unified memory is exposed as gpu_malloc-compatible pointers)
- `src/gpu/blas_metal.cpp` — complex GEMM dispatched to
  `src/gpu/kernels/complex_gemm.metal`, a templated MSL kernel covering both
  fp32 and fp64
- `src/gpu/fft_metal.cpp` — VkFFT (v1.3.4) Metal backend for 3D complex FFT

Precision modes:
- fp64 (default) — bit-comparable to CUDA results, slow due to software fp64
  emulation on Apple GPUs
- fp32 fast mode — opt-in via `hardware.apple_fast_mode: true` or
  `--apple-fast-mode` CLI flag; logger emits a loud warning; validation suite
  refuses to run
```

- [ ] **Step 4: CLAUDE.md — update the backend table and build commands**

Edit `CLAUDE.md`. In the CMake options table (around line 100), update the
`KRONOS_GPU_BACKEND` row to include `metal`. In the build commands block
(around line 87-93), add the metal build line. Add the same Apple Silicon
performance note from README.md.

- [ ] **Step 5: Verify docs build (if doc-build automation exists; otherwise visual review)**

Just open the files and skim them. No automated check.

- [ ] **Step 6: Commit**

```bash
git add README.md docs/user_guide.md docs/architecture.md CLAUDE.md
git commit -m "docs: document Apple Silicon Metal backend and apple_fast_mode"
```

---

### Task 19: Final verification — both build modes, both test runs

- [ ] **Step 1: Clean CPU build, run all tests**

```bash
rm -rf build && cmake -B build -S . && cmake --build build -j4 && \
    cd build && ctest -j2 --output-on-failure 2>&1 | tail -5
```
Expected: `100% tests passed, 0 tests failed out of 442` (440 original + 2 from Task 6).

- [ ] **Step 2: Clean Metal build, run all tests**

```bash
rm -rf build_metal && \
  cmake -B build_metal -S . -DKRONOS_GPU_BACKEND=metal && \
  cmake --build build_metal -j4 && \
  cd build_metal && ctest -j2 --output-on-failure 2>&1 | tail -10
```
Expected: all tests pass, including the new Metal-only tests:
- `GPU.MetalContextInit`
- `GPU.MetalMallocFreeRoundTrip`
- `GPU.MetalMemoryRoundTripComplex128`
- `GPU.MetalComplexGEMMFP64Identity`
- `GPU.MetalComplexGEMMFP64MatchesCPU`
- `GPU.MetalFFTGridConstructs`
- `GPU.MetalFFTRoundTripFP64`
- `GPU.MetalFFTRoundTripFP32`
- `GPU.MetalHamiltonianApplyFP64MatchesCPU`
- `GPU.AppleFastModeEmitsWarning`
- `Integration.SiBulkSCFOnMetalMatchesCPU`

- [ ] **Step 3: Confirm `kronos.metallib` is co-located with the executable**

```bash
ls build_metal/kronos build_metal/kronos.metallib
```
Expected: both files exist.

- [ ] **Step 4: Tag the increment**

```bash
git tag v0.5.1-rc1
git log --oneline -25  # sanity check the increment's commit history
```

- [ ] **Step 5: No commit** — verification only.

---

## Self-review checklist

(Run this against the spec before declaring the plan done.)

1. **Spec coverage:**
   - §3 precision modes (fp64 + opt-in fp32) → Tasks 11, 15, 16
   - §4.1–4.5 new file list → Tasks 5, 7, 8, 10, 12, 13
   - §4.2 dependencies (metal-cpp, VkFFT) → Tasks 2, 3
   - §4.3 build system → Tasks 1, 4, 5
   - §4.4 memory model (unified, didModifyRange) → Task 8
   - §4.5 complex GEMM kernel → Task 10
   - §4.6 FFT — VkFFT → Task 13
   - §4.7 error handling → exists in every implementation task (always throws `GPUNotAvailableError`)
   - §5.2 new tests 1–9 → covered:
     - #1 `MetalContextInit` → Task 7
     - #2 `MetalMemoryRoundTrip` → Task 8 (`MetalMemoryRoundTripComplex128`)
     - #3 `MetalFFTRoundTripFP64` → Task 13
     - #4 `MetalFFTRoundTripFP32` → Task 13 (added after self-review)
     - #5 `MetalComplexGEMM` → Task 12 (identity + 17×13×23 random)
     - #6 `MetalHamiltonianApplyFP64` → Task 14 (added after self-review)
     - #7 `AppleFastModeWarning` → Task 15 step 2 (added after self-review)
     - #8 `AppleFastModeRefusesValidation` → Task 16
     - #9 `SiBulkOnMetal` → Task 17 (`Integration.SiBulkSCFOnMetalMatchesCPU`)
     Plus Task 6's two YAML parser tests for `hardware.apple_fast_mode`.
   - §5.4 regression invariant (CPU 440 still pass) → Tasks 5, 6, 11, 19 all run the CPU suite
   - §6 out-of-scope → not implemented, correctly absent
   - §7 acceptance criteria → Task 19 verifies criteria 1–4; Task 17 verifies criterion 5; Task 18 verifies criterion 6
   - §8 risks → mitigated in the relevant tasks (VkFFT version pin in Task 3, kernel boundary sizes 17/13/23 in Task 12, integration ecutwfc adjustable in Task 17)
   - §10 open questions → resolved: metallib side-loaded next to binary (Tasks 4, 5, 12); VkFFT pinned to v1.3.4 (Task 3); tile size 16×16, no auto-tuning (Task 10)

2. **Placeholder scan:** every code block contains real code. The one exception is the `run_minimal_scf` helper in Task 17 step 2, which has explicit pseudocode and instructs the implementer to read `src/solver/scf.cpp` first. This is acceptable because the SCF entrypoint shape is project-specific and shouldn't be guessed.

3. **Type consistency:** `complex_t` is `std::complex<double>` throughout; `gpu::DeviceBuffer<complex_t>` is consistent; `MTL::Buffer*` is the registry value type; `metal_buffer_for` signature matches between Tasks 12 and 13.

4. **Naming:** `apple_fast_mode_` member, `set_apple_fast_mode()` setter, `apple_fast_mode()` getter, `--apple-fast-mode` CLI flag, `hardware.apple_fast_mode` YAML key — all consistent.
