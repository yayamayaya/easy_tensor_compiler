#include <iostream>
#include "tensor_compiler.hpp"
#include "logging.hpp"

int main()
{
    logger::logger_init("./compiler.log");

    tensor data({2, 4, 3, 3, 1, 5, 1, 1, 1, 1, 1, 1});
    data.set_tensor_size(2, 1, 2, 3);

    _LOG << "Hello log!" << END_;

    std::cout << "Dumping tensor:" <<  data << std::endl;

    _LOG << "Dumping tensor:" << data << END_;

    return 0;
}
