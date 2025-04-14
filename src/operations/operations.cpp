#include "operations.hpp"

tensor 
scalar_add_op::evaluate() const
{
    tensor left_operand = lhs->evaluate();

    return left_operand + rhs;
}