# ============================================================================
# FindSpglib.cmake — Locate the spglib crystal symmetry library
#
# This module defines:
#   Spglib_FOUND          - TRUE if headers and library were found
#   Spglib_INCLUDE_DIRS   - include directories for spglib.h
#   Spglib_LIBRARIES      - libraries to link against
#
# Imported target:
#   Spglib::spglib        - Modern CMake imported target
# ============================================================================

if(TARGET Spglib::spglib)
    set(Spglib_FOUND TRUE)
    return()
endif()

set(_SPGLIB_SEARCH_PATHS
    /usr
    /usr/local
    /opt/local
    /opt/homebrew
    $ENV{SPGLIB_ROOT}
    $ENV{SPGLIB_DIR}
)

find_path(Spglib_INCLUDE_DIR
    NAMES spglib.h
    HINTS ${_SPGLIB_SEARCH_PATHS}
    PATH_SUFFIXES include
)

find_library(Spglib_LIBRARY
    NAMES symspg spglib
    HINTS ${_SPGLIB_SEARCH_PATHS}
    PATH_SUFFIXES lib lib64
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Spglib
    REQUIRED_VARS Spglib_LIBRARY Spglib_INCLUDE_DIR
)

if(Spglib_FOUND)
    set(Spglib_INCLUDE_DIRS ${Spglib_INCLUDE_DIR})
    set(Spglib_LIBRARIES    ${Spglib_LIBRARY})
    mark_as_advanced(Spglib_INCLUDE_DIR Spglib_LIBRARY)

    add_library(Spglib::spglib IMPORTED UNKNOWN)
    set_target_properties(Spglib::spglib PROPERTIES
        IMPORTED_LOCATION             "${Spglib_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${Spglib_INCLUDE_DIR}"
    )
endif()
