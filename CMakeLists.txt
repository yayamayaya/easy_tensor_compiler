cmake_minimum_required(VERSION 3.12)

project(tensor_compiler)

add_definitions(
    -std=c++17 -Wall -Wextra -Wpedantic -Werror -g
)

add_definitions(-DOPTIMIZED_OPERATIONS)

# add_compile_options(-fsanitize=address)
# add_link_options(-fsanitize=address)

option(COMPILE_TESTS "tests compiling" ON)
option(COMPILE_BENCHMARK "benchmark compiling" ON)
option(OPTIMIZED_OPERATIONS "compiling optimizations" ON)

add_subdirectory(src)

if (COMPILE_TESTS)

    find_package(GTest REQUIRED)

    include_directories(${GTEST_INCLUDE_DIRS})

    add_subdirectory(tests)

    target_include_directories(
        tests
        PRIVATE
        ./src/
    )
        
endif ()

if (COMPILE_BENCHMARK AND OPTIMIZED_OPERATIONS)

    # find_package(benchmark REQUIRED)
    add_subdirectory(opt_benchmark)

endif ()

