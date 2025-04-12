#include "operations.hpp"

/*void bin_operation::set_args(const std::vector<Inode *> &args)
{
    arguments = args;
}*/

// bin_operation(const std::shared_ptr<Inode> lhs, const tensor &rhs);

tensor scalar_add_op::evaluate() const
{
    return get_lhs() + get_rhs();
}

