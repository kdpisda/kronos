# ============================================================================
# KRONOS — Kohn-Residual Optimized Numerics Over Silicon
# Root CMake configuration
# ============================================================================
cmake_minimum_required(VERSION 3.20)

project(kronos
    VERSION   0.1.0
    LANGUAGES CXX
)

# ============================================================================
# Global C++ standard
# ============================================================================
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ============================================================================
# Build options
# ============================================================================
set(KRONOS_GPU_BACKEND "none" CACHE STRING
    "GPU acceleration backend: none | cuda | hip | metal")
set_property(CACHE KRONOS_GPU_BACKEND PROPERTY STRINGS none cuda hip metal)

option(KRONOS_BUILD_TESTS  "Build the GoogleTest test suite" ON)
option(KRONOS_BUILD_PYTHON "Build pybind11 Python bindings"  OFF)

# Normalise to lowercase for reliable comparisons throughout the tree.
string(TOLOWER "${KRONOS_GPU_BACKEND}" KRONOS_GPU_BACKEND_LOWER)

# ============================================================================
# Custom find-module path
# ============================================================================
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# ============================================================================
# GPU language enablement (must come before find_package for GPU libs)
# ============================================================================
if(KRONOS_GPU_BACKEND_LOWER STREQUAL "cuda")
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
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
    # Darwin kernel 22.x == macOS 13 (Ventura) — first version with reliable
    # fp64 in Metal Shading Language. CMAKE_SYSTEM_VERSION reports Darwin.
    if(NOT CMAKE_SYSTEM_VERSION VERSION_GREATER_EQUAL "22.0.0")
        message(WARNING
            "KRONOS_GPU_BACKEND=metal is tested on macOS 13+ (Darwin 22+); "
            "current Darwin version ${CMAKE_SYSTEM_VERSION} may have limited "
            "double-precision Metal Shading Language support.")
    endif()
endif()

# ============================================================================
# Required dependencies
# ============================================================================
find_package(FFTW3   REQUIRED)
find_package(BLAS    REQUIRED)
find_package(LAPACK  REQUIRED)
find_package(yaml-cpp REQUIRED)

# ============================================================================
# Optional dependencies
# ============================================================================
find_package(HDF5 COMPONENTS C)
find_package(MPI  COMPONENTS CXX)
find_package(LibXC)
find_package(Spglib)

# ============================================================================
# GPU-specific dependencies
# ============================================================================
if(KRONOS_GPU_BACKEND_LOWER STREQUAL "cuda")
    find_package(CUDAToolkit REQUIRED)
    message(STATUS "KRONOS: CUDA backend enabled (toolkit ${CUDAToolkit_VERSION})")

elseif(KRONOS_GPU_BACKEND_LOWER STREQUAL "hip")
    # ROCm / HIP discovery
    find_package(hip      REQUIRED CONFIG)
    find_package(rocfft   REQUIRED CONFIG)
    find_package(rocblas  REQUIRED CONFIG)
    message(STATUS "KRONOS: HIP backend enabled")

elseif(KRONOS_GPU_BACKEND_LOWER STREQUAL "metal")
    # metal-cpp: Apple's official C++17 headers for Metal (header-only).
    #
    # NOTE on URL stability: Apple rotates metal-cpp filenames with each SDK
    # release and does NOT maintain permanent links. If the URL below 404s,
    # visit https://developer.apple.com/metal/cpp/ and pick the closest
    # successor, then recompute SHA-256 with:
    #     shasum -a 256 metal-cpp_macOSXX.Y_iOSAA.B.zip
    include(FetchContent)
    FetchContent_Declare(
        metal_cpp
        URL      https://developer.apple.com/metal/cpp/files/metal-cpp_macOS15.2_iOS18.2.zip
        URL_HASH SHA256=3437e4abfbd3d45217f34772ef3502f31ba3358e5fb6ac9d0ca952a047bcfe25
    )
    FetchContent_MakeAvailable(metal_cpp)
    # The macOS15+ archive extracts flat (Metal/ Foundation/ QuartzCore/ directly
    # under the source root), so Metal/Metal.hpp is at ${metal_cpp_SOURCE_DIR}/Metal/Metal.hpp.
    #
    # The INTERFACE-library target is named kronos_metal_cpp to avoid colliding
    # with the FetchContent population name (metal_cpp) — downstream callers
    # `target_link_libraries(... kronos_metal_cpp)`.
    add_library(kronos_metal_cpp INTERFACE)
    target_include_directories(kronos_metal_cpp INTERFACE "${metal_cpp_SOURCE_DIR}")
    message(STATUS "KRONOS: Metal backend enabled (metal-cpp at ${metal_cpp_SOURCE_DIR})")

    # VkFFT: portable GPU FFT (uses Metal via metal-cpp under the hood).
    # Pinned to v1.3.4 — newer versions may change the Metal-backend API.
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
    # FetchContent population name (vkfft) — downstream callers use kronos_vkfft.
    add_library(kronos_vkfft INTERFACE)
    target_include_directories(kronos_vkfft INTERFACE ${vkfft_SOURCE_DIR}/vkFFT)
    target_compile_definitions(kronos_vkfft INTERFACE VKFFT_BACKEND=5)  # 5 = Metal
    message(STATUS "KRONOS: VkFFT fetched at ${vkfft_SOURCE_DIR}")
endif()

# ============================================================================
# Subdirectories
# ============================================================================
add_subdirectory(src)

if(KRONOS_BUILD_TESTS)
    enable_testing()
    add_subdirectory(test)
endif()

# ============================================================================
# Summary
# ============================================================================
message(STATUS "")
message(STATUS "──── KRONOS ${PROJECT_VERSION} configuration ────")
message(STATUS "  C++ standard   : C++${CMAKE_CXX_STANDARD}")
message(STATUS "  GPU backend    : ${KRONOS_GPU_BACKEND}")
message(STATUS "  Build tests    : ${KRONOS_BUILD_TESTS}")
message(STATUS "  Build Python   : ${KRONOS_BUILD_PYTHON}")
message(STATUS "  HDF5 found     : ${HDF5_FOUND}")
message(STATUS "  MPI found      : ${MPI_FOUND}")
message(STATUS "  LibXC found    : ${LibXC_FOUND}")
message(STATUS "  Spglib found   : ${Spglib_FOUND}")
message(STATUS "────────────────────────────────────────────")
message(STATUS "")
