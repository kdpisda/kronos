# ============================================================================
# FindFFTW3.cmake — Locate the FFTW3 double-precision library
#
# This module defines:
#   FFTW3_FOUND          - TRUE if headers and library were found
#   FFTW3_INCLUDE_DIRS   - include directories for fftw3.h
#   FFTW3_LIBRARIES      - libraries to link against
#
# Imported target:
#   FFTW3::fftw3          - Modern CMake imported target
# ============================================================================

# Guard against repeated inclusion.
if(TARGET FFTW3::fftw3)
    set(FFTW3_FOUND TRUE)
    return()
endif()

# ============================================================================
# Search hints — common install prefixes
# ============================================================================
set(_FFTW3_SEARCH_PATHS
    /usr
    /usr/local
    /opt/local          # MacPorts
    /opt/homebrew       # Homebrew on Apple Silicon
    /usr/local/Cellar/fftw   # older Homebrew
    $ENV{FFTW_ROOT}
    $ENV{FFTW3_ROOT}
    $ENV{FFTW_DIR}
)

# ============================================================================
# Find header
# ============================================================================
find_path(FFTW3_INCLUDE_DIR
    NAMES fftw3.h
    HINTS ${_FFTW3_SEARCH_PATHS}
    PATH_SUFFIXES include
)

# ============================================================================
# Find library (double-precision: libfftw3)
# ============================================================================
find_library(FFTW3_LIBRARY
    NAMES fftw3
    HINTS ${_FFTW3_SEARCH_PATHS}
    PATH_SUFFIXES lib lib64
)

# ============================================================================
# Standard handling (sets FFTW3_FOUND, handles REQUIRED / QUIET)
# ============================================================================
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(FFTW3
    REQUIRED_VARS FFTW3_LIBRARY FFTW3_INCLUDE_DIR
)

# ============================================================================
# Publish results & create imported target
# ============================================================================
if(FFTW3_FOUND)
    set(FFTW3_INCLUDE_DIRS ${FFTW3_INCLUDE_DIR})
    set(FFTW3_LIBRARIES    ${FFTW3_LIBRARY})
    mark_as_advanced(FFTW3_INCLUDE_DIR FFTW3_LIBRARY)

    add_library(FFTW3::fftw3 IMPORTED UNKNOWN)
    set_target_properties(FFTW3::fftw3 PROPERTIES
        IMPORTED_LOCATION             "${FFTW3_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${FFTW3_INCLUDE_DIR}"
    )
endif()
