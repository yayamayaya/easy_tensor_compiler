#ifndef ETC_OPERATIONS_H
#define ETC_OPERATIONS_H

#include "tensor.hpp"
#include <memory>

class node_interface
{
public:
    virtual ~node_interface() {};

    virtual tensor evaluate() const =0;  
};

// class operation_interface: public node_interface
// {
// public:
//     virtual ~operation_interface() {};



//     // virtual void set_args(const std::vector<node_interface *> &args) = 0;
//     // virtual const std::vector<node_interface *> &get_args()    const = 0;
// };

class input_data: public node_interface
{
private:
    tensor node_data;

public:
    input_data(const tensor &t): node_data(t) {};

    tensor evaluate() const override { return node_data; }
};

class bin_operation: public node_interface
{
    // std::vector<Inode *> args;
public:

    std::shared_ptr<node_interface> lhs;
    const tensor&                   rhs;


    bin_operation() =delete;

    bin_operation(std::shared_ptr<node_interface> lhs_val, const tensor &rhs_val): lhs(lhs_val), rhs(rhs_val) {};  

    // void bin_operation::set_args(const std::vector<node_interface *> &args) override;

    // const std::vector<node_interface *> &get_args() const override { return {lhs.get(), rhs}; }
};

class scalar_add_op: public bin_operation {tensor evaluate() const override;};

class scalar_sub_op: public bin_operation {tensor evaluate() const override;};

class scalar_mul_op: public bin_operation {tensor evaluate() const override;};

class matrix_mul_op: public bin_operation {tensor evaluate() const override;};

class mat_convol_op: public bin_operation {tensor evaluate() const override;};

class un_operation:  public node_interface
{
public:
    un_operation(const std::shared_ptr<node_interface> arg);
};

class relu_op:    public node_interface {};

class softmax_op: public node_interface {};

#endif