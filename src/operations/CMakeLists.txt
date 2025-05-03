
add_library(operations operations.cpp operations.hpp)

target_link_libraries(operations tensor_lib)

target_include_directories(operations PRIVATE ./../tensor_class)