include(FetchContent)

# Pinned to a tag, not a branch: a test framework that changes under the project
# turns an unrelated commit into a red build.
FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.7.1
    GIT_SHALLOW TRUE)
FetchContent_MakeAvailable(Catch2)

add_executable(tinytensor_tests
    ops_test.cpp
    shape_test.cpp
    tensor_test.cpp)

target_link_libraries(tinytensor_tests PRIVATE
    tinytensor::tinytensor
    Catch2::Catch2WithMain)

tinytensor_set_warnings(tinytensor_tests)

# Registers every TEST_CASE with CTest individually, so a failure names itself
# instead of reporting "the test binary failed".
list(APPEND CMAKE_MODULE_PATH "${catch2_SOURCE_DIR}/extras")
include(Catch)
catch_discover_tests(tinytensor_tests)
