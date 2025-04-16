
add_library(optimizations optimized_operations.cpp)

target_link_libraries(optimizations tensor_lib)

target_include_directories(optimizations PRIVATE ./../tensor_class)