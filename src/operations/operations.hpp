#ifndef ETC_OPERATIONS_H
#define ETC_OPERATIONS_H

#include "tensor.hpp"
#include <memory>

class node
{
public:
    virtual ~node() {};

    virtual tensor evaluate() const =0;  
};

class operation: public node
{
public:
    virtual ~operation() {};

    virtual void set_args(const std::vector<node *> &args) = 0;
    virtual const std::vector<node *> &get_args()    const = 0;
};

class input_data: public node
{
private:
    tensor node_data;

public:
    input_data(const tensor &t): node_data(t) {};

    tensor evaluate() const override
    {
        return node_data;
    }
};

class bin_operation: public operation
{
    // std::vector<Inode *> args;

    tensor lhs;

    tensor rhs;

public:
    // void bin_operation::set_args(const std::vector<Inode *> &args) override;

    tensor get_lhs() const
    {
        return lhs;
    }

    tensor get_rhs() const
    {
        return rhs;
    }

    /*const std::vector<Inode *> &get_args() const override
    {
        return args;
    }*/

    bin_operation(const std::shared_ptr<node> lhs_val, const tensor &rhs_val): lhs(lhs_val->evaluate()), rhs(rhs_val) {};  
};

class scalar_add_op: public bin_operation {tensor evaluate() const override;};

class scalar_sub_op: public bin_operation {tensor evaluate() const override;};

class scalar_mul_op: public bin_operation {tensor evaluate() const override;};

class matrix_mul_op: public bin_operation {tensor evaluate() const override;};

class mat_convol_op: public bin_operation {tensor evaluate() const override;};

class un_operation:  public operation
{
public:
    un_operation(const std::shared_ptr<node> arg);
};

class relu_op:    public node {};
class softmax_op: public node {};

#endif