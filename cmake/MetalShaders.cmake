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
