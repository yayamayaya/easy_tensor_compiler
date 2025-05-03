
add_executable(bench benchmarking.cpp benchmarking.hpp)

find_package(benchmark REQUIRED)

target_link_libraries(bench benchmark::benchmark tensor_lib)

target_include_directories(bench PRIVATE ./../src/tensor_class)