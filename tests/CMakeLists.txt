
add_executable(tests tests.hpp tests.cpp)

target_link_libraries(tests tensor_compiler -lgtest)