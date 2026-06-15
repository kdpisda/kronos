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
#
# ----------------------------------------------------------------------------
# PREREQUISITE — full Xcode.app, not just Command Line Tools
# ----------------------------------------------------------------------------
# `xcrun metal` (the Metal Shading Language compiler) ships ONLY with the full
# Xcode app from the App Store (~16 GB). It is NOT included in Xcode Command
# Line Tools. If this build fails with:
#     xcrun: error: unable to find utility "metal", not a developer tool or in PATH
# install Xcode.app, then run:
#     sudo xcode-select --switch /Applications/Xcode.app
#
# The KRONOS Metal backend work was paused at Task 5 (skeleton) on 2026-05-17
# specifically because of this dependency. To resume:
#   - docs/superpowers/specs/2026-05-16-apple-silicon-metal-backend-design.md
#   - docs/superpowers/plans/2026-05-16-apple-silicon-metal-backend.md
# Pick up from Task 6 onward once Xcode.app is available.
# ============================================================================

function(kronos_build_metallib)
    cmake_parse_arguments(KMM "" "TARGET;OUTPUT;SDK" "SOURCES" ${ARGN})

    # Fail fast on missing required args — keeps configure-time errors close
    # to the call site instead of producing a silent empty .metallib later.
    if(NOT KMM_TARGET)
        message(FATAL_ERROR "kronos_build_metallib: TARGET is required")
    endif()
    if(NOT KMM_OUTPUT)
        message(FATAL_ERROR "kronos_build_metallib: OUTPUT is required")
    endif()
    if(NOT KMM_SOURCES)
        message(FATAL_ERROR "kronos_build_metallib: SOURCES is required (at least one .metal file)")
    endif()
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
