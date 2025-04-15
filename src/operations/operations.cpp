#include "operations.hpp"
#include <cmath>
#include <iostream>

tensor 
scalar_add_op::evaluate() const
{
    return lhs->evaluate() + rhs;
}

tensor
scalar_sub_op::evaluate() const
{
    return lhs->evaluate() - rhs;
}

tensor
matrix_mul_op::evaluate() const
{
    return lhs->evaluate() * rhs;
}

tensor
mat_convol_op::evaluate() const
{
    return lhs->evaluate() / rhs;
}

tensor
scalar_mul_op::evaluate() const
{
    return lhs->evaluate() * rhs;
}

tensor
relu_op::evaluate() const
{
    tensor operand = lhs->evaluate();

    std::vector<number_t> res_data;

    for (index_t i = 0; i < operand.data.size(); i++)
    {
        if (operand.data[i] < 0)
            res_data.push_back(0);
        else
            res_data.push_back(operand.data[i]);
    }

    tensor res(res_data);
    res.set_tensor_size(operand.size);

    return res;
}

tensor
softmax_op::evaluate() const
{
    tensor operand = lhs->evaluate();

    tensor result{std::vector<number_t>()};
    result.set_tensor_size(operand.size);
        
    for (index_t i = 0; i < operand.size.N; i++)
    {
        std::vector<number_t> denominators;
        
        for (index_t j = 0; j < operand.size.H; j++)
            for (index_t k = 0; k < operand.size.W; k++)
            {
                number_t den = 0;

                for (index_t r = 0; r < operand.size.C; r++)
                    den += std::exp(operand(i, r, j, k));
                
                for (index_t r = 0; r < operand.size.C; r++)
                    result(i, r, j, k) = std::exp(operand(i, r, j, k)) / den;

            }
    }
        
    return result;
}

