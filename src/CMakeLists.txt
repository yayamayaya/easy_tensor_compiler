

add_subdirectory(tensor_class)

add_subdirectory(operations)

add_subdirectory(optimizations)

add_library(tensor_compiler tensor_compiler.hpp tensor_compiler.cpp)

target_include_directories(
    tensor_compiler
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/operations/
    ${CMAKE_CURRENT_SOURCE_DIR}/tensor_class/
)

target_link_libraries(tensor_compiler PRIVATE operations)
