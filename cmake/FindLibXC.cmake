# ============================================================================
# FindLibXC.cmake — Locate the libxc exchange-correlation library
#
# This module defines:
#   LibXC_FOUND          - TRUE if headers and library were found
#   LibXC_INCLUDE_DIRS   - include directories for xc.h
#   LibXC_LIBRARIES      - libraries to link against
#   LibXC_VERSION        - version string (if discoverable)
#
# Imported target:
#   LibXC::xc            - Modern CMake imported target
# ============================================================================

# Guard against repeated inclusion.
if(TARGET LibXC::xc)
    set(LibXC_FOUND TRUE)
    return()
endif()

# ============================================================================
# Search hints — common install prefixes
# ============================================================================
set(_LIBXC_SEARCH_PATHS
    /usr
    /usr/local
    /opt/local
    /opt/homebrew
    $ENV{LIBXC_ROOT}
    $ENV{LIBXC_DIR}
    $ENV{XC_ROOT}
)

# ============================================================================
# Find header
# ============================================================================
find_path(LibXC_INCLUDE_DIR
    NAMES xc.h
    HINTS ${_LIBXC_SEARCH_PATHS}
    PATH_SUFFIXES include
)

# ============================================================================
# Find library
# ============================================================================
find_library(LibXC_LIBRARY
    NAMES xc
    HINTS ${_LIBXC_SEARCH_PATHS}
    PATH_SUFFIXES lib lib64
)

# ============================================================================
# Try to extract the version from the header
# ============================================================================
if(LibXC_INCLUDE_DIR AND EXISTS "${LibXC_INCLUDE_DIR}/xc_version.h")
    file(STRINGS "${LibXC_INCLUDE_DIR}/xc_version.h" _xc_ver_major
         REGEX "#define[ \t]+XC_MAJOR_VERSION[ \t]+")
    file(STRINGS "${LibXC_INCLUDE_DIR}/xc_version.h" _xc_ver_minor
         REGEX "#define[ \t]+XC_MINOR_VERSION[ \t]+")
    file(STRINGS "${LibXC_INCLUDE_DIR}/xc_version.h" _xc_ver_micro
         REGEX "#define[ \t]+XC_MICRO_VERSION[ \t]+")

    if(_xc_ver_major AND _xc_ver_minor AND _xc_ver_micro)
        string(REGEX REPLACE ".*XC_MAJOR_VERSION[ \t]+([0-9]+).*" "\\1"
               _xc_major "${_xc_ver_major}")
        string(REGEX REPLACE ".*XC_MINOR_VERSION[ \t]+([0-9]+).*" "\\1"
               _xc_minor "${_xc_ver_minor}")
        string(REGEX REPLACE ".*XC_MICRO_VERSION[ \t]+([0-9]+).*" "\\1"
               _xc_micro "${_xc_ver_micro}")
        set(LibXC_VERSION "${_xc_major}.${_xc_minor}.${_xc_micro}")
    endif()
    unset(_xc_ver_major)
    unset(_xc_ver_minor)
    unset(_xc_ver_micro)
    unset(_xc_major)
    unset(_xc_minor)
    unset(_xc_micro)
endif()

# ============================================================================
# Standard handling
# ============================================================================
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(LibXC
    REQUIRED_VARS LibXC_LIBRARY LibXC_INCLUDE_DIR
    VERSION_VAR   LibXC_VERSION
)

# ============================================================================
# Publish results & create imported target
# ============================================================================
if(LibXC_FOUND)
    set(LibXC_INCLUDE_DIRS ${LibXC_INCLUDE_DIR})
    set(LibXC_LIBRARIES    ${LibXC_LIBRARY})
    mark_as_advanced(LibXC_INCLUDE_DIR LibXC_LIBRARY)

    add_library(LibXC::xc IMPORTED UNKNOWN)
    set_target_properties(LibXC::xc PROPERTIES
        IMPORTED_LOCATION             "${LibXC_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${LibXC_INCLUDE_DIR}"
    )
endif()
