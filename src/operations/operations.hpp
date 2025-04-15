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
public:
    const std::shared_ptr<node_interface> lhs;
    const tensor&                         rhs;

    bin_operation() =delete;

    bin_operation(const std::shared_ptr<node_interface> lhs_val, const tensor &rhs_val): lhs(lhs_val), rhs(rhs_val) {};

    ~bin_operation() {};
};

class scalar_add_op: public bin_operation 
{
public: 
    scalar_add_op(const std::shared_ptr<node_interface> lhs_val, const tensor &rhs_val): bin_operation(lhs_val, rhs_val) {}; 
    tensor evaluate() const override;
};

class scalar_sub_op: public bin_operation 
{
public: 
    scalar_sub_op(const std::shared_ptr<node_interface> lhs_val, const tensor &rhs_val): bin_operation(lhs_val, rhs_val) {}; 
    tensor evaluate() const override;
};

class matrix_mul_op: public bin_operation 
{
public: 
    matrix_mul_op(const std::shared_ptr<node_interface> lhs_val, const tensor &rhs_val): bin_operation(lhs_val, rhs_val) {}; 
    tensor evaluate() const override;
};

class mat_convol_op: public bin_operation 
{
public:
    mat_convol_op(const std::shared_ptr<node_interface> lhs_val, const tensor &rhs_val): bin_operation(lhs_val, rhs_val) {}; 
    tensor evaluate() const override;
};

class scalar_mul_op: public node_interface 
{
    const std::shared_ptr<node_interface> lhs;
    const number_t&                       rhs;

    tensor evaluate() const override;

public:

    scalar_mul_op() =delete;

    scalar_mul_op(const std::shared_ptr<node_interface> lhs_val, const number_t &rhs_val): lhs(lhs_val), rhs(rhs_val) {};
};

class un_operation:  public node_interface
{
public:

    const std::shared_ptr<node_interface> lhs;

    un_operation(const std::shared_ptr<node_interface> arg): lhs(arg) {};

    ~un_operation() {};
};

class relu_op:    public un_operation 
{
public:
    relu_op(const std::shared_ptr<node_interface> arg): un_operation(arg) {};
    tensor evaluate() const override;
};

class softmax_op: public un_operation 
{
public:
    softmax_op(const std::shared_ptr<node_interface> arg): un_operation(arg) {};
    tensor evaluate() const override;
};

#endif