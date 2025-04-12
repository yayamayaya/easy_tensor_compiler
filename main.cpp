#include <iostream>
#include "tensor_compiler.hpp"
#include "logging.hpp"

int main(int argc, char const *argv[])
{
    logger::logger_init("./compiler.log");

    tensor data{2, 4, 3, 3};

    _LOG << "Hello log!" << END_;

    std::vector<number_t> vec = {};
    for (int i = 0; i < 2*4*3*3 ; i++)
    {
        vec.push_back(i);
    }
    
    data = vec;

    std::cout << "Dumping tensor:" <<  data << std::endl;

    _LOG << "Dumping tensor:" << data << END_;

    return 0;
}
