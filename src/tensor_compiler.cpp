#include "tensor_compiler.hpp"

std::shared_ptr<node_interface>
neural_network::add_op(std::shared_ptr<node_interface> op)
{
    nodes.push_back(op);

    return op;
}

tensor
neural_network::infer()
{
    for (index_t i = 0; i < nodes.size(); i++)
        nodes[i] = std::make_shared<input_data>(nodes[i]->evaluate());

    return nodes.back()->evaluate();
}