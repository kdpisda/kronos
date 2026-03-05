# ============================================================================
# KRONOS  test/CMakeLists.txt
# Unit tests built with GoogleTest (fetched automatically).
# ============================================================================

include(FetchContent)

FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.14.0
)
# Prevent GoogleTest from overriding our compiler/linker settings on Windows.
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

include(GoogleTest)

# ============================================================================
# Helper: add a KRONOS test in one line
# ============================================================================
function(kronos_add_test TEST_NAME)
    add_executable(${TEST_NAME} ${TEST_NAME}.cpp)
    target_link_libraries(${TEST_NAME}
        PRIVATE
            kronos_lib
            GTest::gtest_main
    )
    gtest_discover_tests(${TEST_NAME}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )
endfunction()

# ============================================================================
# Register test executables
# ============================================================================
kronos_add_test(test_input)
kronos_add_test(test_fft)
kronos_add_test(test_basis)
kronos_add_test(test_upf)
kronos_add_test(test_solvers)
kronos_add_test(test_physics)
kronos_add_test(test_scf)
kronos_add_test(test_crystal)
kronos_add_test(test_hamiltonian)
kronos_add_test(test_postprocessing)
kronos_add_test(test_output)
kronos_add_test(test_gradient)
kronos_add_test(test_forces)
kronos_add_test(test_utils)
kronos_add_test(test_unimplemented)
