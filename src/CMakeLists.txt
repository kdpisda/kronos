# ============================================================================
# KRONOS  src/CMakeLists.txt
# Builds the core static library and the main executable.
# ============================================================================

# ============================================================================
# Collect CPU sources via GLOB (re-run CMake when adding new files)
# ============================================================================
file(GLOB_RECURSE KRONOS_CORE_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/core/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/basis/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/io/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/potential/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/solver/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/hamiltonian/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/postprocessing/*.cpp
)

# ============================================================================
# Static library: kronos_lib
# ============================================================================
add_library(kronos_lib STATIC ${KRONOS_CORE_SOURCES})

target_include_directories(kronos_lib
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src
)

# -- Required link dependencies (modern imported targets where available) -----
target_link_libraries(kronos_lib
    PUBLIC
        FFTW3::fftw3
        ${BLAS_LIBRARIES}
        ${LAPACK_LIBRARIES}
        yaml-cpp::yaml-cpp
)

# -- Optional: HDF5 ----------------------------------------------------------
if(HDF5_FOUND)
    target_compile_definitions(kronos_lib PUBLIC KRONOS_HAS_HDF5)
    target_include_directories(kronos_lib PUBLIC ${HDF5_INCLUDE_DIRS})
    target_link_libraries(kronos_lib PUBLIC ${HDF5_LIBRARIES})
endif()

# -- Optional: MPI -----------------------------------------------------------
if(MPI_FOUND)
    target_compile_definitions(kronos_lib PUBLIC KRONOS_HAS_MPI)
    target_link_libraries(kronos_lib PUBLIC MPI::MPI_CXX)
endif()

# -- Optional: LibXC ---------------------------------------------------------
if(LibXC_FOUND)
    target_compile_definitions(kronos_lib PUBLIC KRONOS_HAS_LIBXC)
    target_link_libraries(kronos_lib PUBLIC LibXC::xc)
endif()

# -- Optional: Spglib -------------------------------------------------------
if(Spglib_FOUND)
    target_compile_definitions(kronos_lib PUBLIC KRONOS_HAS_SPGLIB)
    target_link_libraries(kronos_lib PUBLIC Spglib::spglib)
endif()

# ============================================================================
# GPU sources & linking
# ============================================================================
if(KRONOS_GPU_BACKEND_LOWER STREQUAL "cuda")
    file(GLOB_RECURSE KRONOS_GPU_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/gpu/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/gpu/*.cu
    )
    target_sources(kronos_lib PRIVATE ${KRONOS_GPU_SOURCES})
    target_compile_definitions(kronos_lib PUBLIC KRONOS_GPU_CUDA)
    target_link_libraries(kronos_lib
        PUBLIC
            CUDA::cufft
            CUDA::cublas
    )

elseif(KRONOS_GPU_BACKEND_LOWER STREQUAL "hip")
    file(GLOB_RECURSE KRONOS_GPU_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/gpu/*.cpp
    )
    target_sources(kronos_lib PRIVATE ${KRONOS_GPU_SOURCES})
    target_compile_definitions(kronos_lib PUBLIC KRONOS_GPU_HIP)
    target_link_libraries(kronos_lib
        PUBLIC
            hip::host
            roc::rocfft
            roc::rocblas
    )

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

    # Compile .metal shaders into kronos.metallib (placed at the build root)
    kronos_build_metallib(
        TARGET  kronos_metallib
        OUTPUT  ${CMAKE_BINARY_DIR}/kronos.metallib
        SDK     macosx
        SOURCES gpu/kernels/complex_gemm.metal
    )
    add_dependencies(kronos_lib kronos_metallib)

else()
    # CPU-only build: compile GPU stubs so callers get clean runtime errors
    target_sources(kronos_lib PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/gpu/gpu_stubs.cpp
    )
endif()

# ============================================================================
# Main executable
# ============================================================================
add_executable(kronos main.cpp)
target_link_libraries(kronos PRIVATE kronos_lib)
