

add_library(tensor_lib tensor.cpp tensor.hpp)

if (OPTIMIZED_OPERATIONS)

    target_link_libraries(tensor_lib PRIVATE opt)

    target_include_directories(tensor_lib PRIVATE ./optimizations/)

    add_subdirectory(optimizations)

endif()