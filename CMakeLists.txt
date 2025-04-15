cmake_minimum_required(VERSION 3.12)

project(tensor_compiler)

add_definitions(
    -std=c++17 -Wall -Wextra -Wpedantic -Werror
)

add_subdirectory(src)

option(COMPILE_TESTS "tests compiling" ON)

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


