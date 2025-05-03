

add_library(opt optimized_operations.cpp)

target_link_libraries(opt tensor_lib)

target_include_directories(opt PRIVATE ./..)