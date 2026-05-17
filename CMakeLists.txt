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
