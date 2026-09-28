add_library(tinytensor
    src/ops.cpp
    src/shape.cpp
    src/tensor.cpp)

add_library(tinytensor::tinytensor ALIAS tinytensor)

# PUBLIC: anything linking the library sees the headers. The sources are found
# through the same path, so an include reads `<tinytensor/shape.hpp>` whether it
# is inside the library or outside it.
target_include_directories(tinytensor PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

target_compile_features(tinytensor PUBLIC cxx_std_20)

tinytensor_set_warnings(tinytensor)
