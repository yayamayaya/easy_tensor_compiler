#ifndef TESTS_H
#define TESTS_H

#include <gtest/gtest.h>
#include "tensor_compiler.hpp"

TEST(tensor_testing, tensor_print)
{
    tensor t({1, 1, 2, 2});
    t.set_tensor_size(1, 1, 2, 2);

    tensor t2({1, 2, 3, 4, 4, 4, 4, 5, 61, 21, 2, 3, 10, 9, 8, 7});
    t2.set_tensor_size(2, 4, 1, 2);

    std::cout << t << std::endl;
}

TEST(tensor_testing, index_test)
{
    tensor t1({1, 1, 1});
    t1.set_tensor_size(1, 1, 3, 1);
    EXPECT_NO_THROW(t1(0, 0, 1, 0));

    tensor t2({1, 1, 2, 2});
    t2.set_tensor_size(1, 1, 2, 2);
    EXPECT_ANY_THROW(t2(2, 2, 2, 2));

    tensor t3({2, 1, 4, 3});
    EXPECT_ANY_THROW(t2(1, 1, 1, 1));
}

TEST(tensor_testing, scalar_add)
{
    tensor r({1, 2, 3});
    r.set_tensor_size(1, 1, 1, 3);

    tensor l({3, 2, 1});
    l.set_tensor_size(1, 1, 1, 3);

    tensor res({4, 4, 4});
    res.set_tensor_size(1, 1, 1, 3);

    EXPECT_EQ(r + l, res);

    tensor r2({1, 1, 1});
    r2.set_tensor_size(1, 1, 1, 3);

    tensor l2({1, 1, 1});
    l2.set_tensor_size(1, 1, 3, 1);

    EXPECT_ANY_THROW(l2 + r2);
}

TEST(tensor_testing, scalar_mul)
{
    tensor r({1, 2, 3});
    r.set_tensor_size(1, 1, 3, 1);

    tensor res({2, 4, 6});
    res.set_tensor_size(1, 1, 3, 1);

    EXPECT_EQ(r * 2, res);
}

TEST(tensor_testing, mat_mul)
{
    tensor l1({1, 2, 3});
    l1.set_tensor_size(1, 1, 1, 3);

    tensor r1({4, 5, 6});
    r1.set_tensor_size(1, 1, 3, 1);

    tensor res({32});
    res.set_tensor_size(1, 1, 1, 1);

    EXPECT_EQ(l1 * r1, res);

    tensor l2({1, 2, 3});
    l2.set_tensor_size(1, 1, 1, 3);

    tensor r2({1, 2, 3});
    r2.set_tensor_size(1, 1, 1, 3);

    EXPECT_ANY_THROW(l2 * r2);
}

TEST(operation_testing, bin_operation_test)
{
    tensor t1({1, 2, 3});
    t1.set_tensor_size(1, 1, 1, 3);

    tensor t2({3, 2, 1});
    t2.set_tensor_size(1, 1, 1, 3);

    std::shared_ptr<node_interface> input = std::make_shared<input_data>(t1);
    std::shared_ptr<scalar_add_op> op     = std::make_shared<scalar_add_op>(input, t2);

    tensor res({4, 4, 4});
    res.set_tensor_size(1, 1, 1, 3);

    EXPECT_EQ(op->evaluate(), res);
}

TEST(operation_testing, convol_test)
{
    tensor t1({1, 2, 3, 4, 5, 6, 7, 8, 9});
    t1.set_tensor_size(1, 1, 3, 3);

    tensor t2({-1, 0, 1, 2});
    t2.set_tensor_size(1, 1, 2, 2);

    std::shared_ptr<input_data> input = std::make_shared<input_data>(t1);
    std::shared_ptr<mat_convol_op> op = std::make_shared<mat_convol_op>(input, t2);

    tensor res({13, 15, 19, 21});
    res.set_tensor_size(1, 1, 2, 2);

    EXPECT_EQ(op->evaluate(), res);
}

TEST(operation_testing, relu_test)
{
    tensor t1({1, 2, -1, 5, -6, -9});
    t1.set_tensor_size(1, 1, 3, 2);

    std::shared_ptr<input_data> input = std::make_shared<input_data>(t1);
    std::shared_ptr<relu_op> op       = std::make_shared<relu_op>(input);

    tensor t2({1, 2, 0, 5, 0, 0});
    t2.set_tensor_size(1, 1, 3, 2);

    EXPECT_EQ(op->evaluate(), t2);
}

TEST(operation_testing, softmax_test)
{
    tensor t1({2, 5, 3, 2, 3, 66, 7, 1});
    t1.set_tensor_size(1, 2, 2, 2);

    tensor res({0.2689, 0, 0.018, 0.7311, 0.7311, 1, 0.982, 0.2689});
    res.set_tensor_size(1, 2, 2, 2);

    std::shared_ptr<input_data>input = std::make_shared<input_data>(t1);
    std::shared_ptr<softmax_op>op    = std::make_shared<softmax_op>(input);

    EXPECT_NO_THROW(op->evaluate());

    std::cout << "these values must be equal: " << op->evaluate() << " and " << res << std::endl;
}

TEST(neural_network_testing, simple_network)
{
    tensor in({6, 3, 2, 6, 7, 1, -6, 1, 6, 2, 1, 6, 8, 3, 1, 5, 80, 12});
    in.set_tensor_size(2, 1, 3, 3);

    std::shared_ptr<input_data>input = std::make_shared<input_data>(in);

    tensor weight({1, 5, 3, 6, 8, 9, 2, 3, 5, 7, 2, 1, 5, 6, 7, 8, 3, 111});
    weight.set_tensor_size(2, 1, 3, 3);

    neural_network nn;
    nn.add_op(std::make_shared<matrix_mul_op>(input, weight));

    EXPECT_NO_THROW(nn.infer());
}

TEST(neural_network_testing, big_network)
{
    tensor in2({1, 2, 61, 132, 1, 5});
    in2.set_tensor_size(1, 1, 3, 2);

    number_t w1 = 1.2;

    tensor w2({5, 1, 6, 1, 2, 6});
    w2.set_tensor_size(1, 1, 2, 3);

    tensor w3({1, 2, 3, 4});
    w3.set_tensor_size(1, 1, 2, 2);

    tensor w4({4});
    w4.set_tensor_size(1, 1, 1, 1);
    
    std::shared_ptr<input_data>input1 = std::make_shared<input_data>(in2);
    std::shared_ptr<scalar_mul_op>op1 = std::make_shared<scalar_mul_op>(input1, w1);
    std::shared_ptr<matrix_mul_op>op2 = std::make_shared<matrix_mul_op>(op1, w2);
    std::shared_ptr<mat_convol_op>op3 = std::make_shared<mat_convol_op>(op2, w3);
    std::shared_ptr<mat_convol_op>op4 = std::make_shared<mat_convol_op>(op3, w4);
    
    neural_network nn;

    nn.add_op(op1);
    nn.add_op(op2);
    nn.add_op(op3);
    nn.add_op(op4);

    EXPECT_NO_THROW(nn.infer());
}


#endif