#ifndef TENSOR_COMP_H
#define TENSOR_COMP_H

#include <memory>
#include "operations.hpp" 
#include "tensor.hpp"

class neural_network
{
    std::vector<std::shared_ptr<node_interface>> nodes;

public:

    neural_network(): nodes() {};

    ~neural_network() {};

    std::shared_ptr<node_interface> add_op(std::shared_ptr<node_interface> op);

    tensor infer();
};

#endif