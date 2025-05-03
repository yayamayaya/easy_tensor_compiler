#ifndef TESTS_H
#define TESTS_H

#include <gtest/gtest.h>
#include "tensor_compiler.hpp"

TEST(tensor_testing, tensor_print)
{
    tensor t(1, 1, 2, 2, {1, 1, 2, 2});

    tensor t2(2, 4, 1, 2, {1, 2, 3, 4, 4, 4, 4, 5, 61, 21, 2, 3, 10, 9, 8, 7});

    std::cout << t << std::endl;
}

TEST(tensor_testing, index_test)
{
    tensor t1(1, 1, 3, 1, {1, 1, 1});
    EXPECT_NO_THROW(t1(0, 0, 1, 0));

    tensor t2(1, 1, 2, 2, {1, 1, 2, 2});
    EXPECT_ANY_THROW(t2(2, 2, 2, 2));
}

TEST(tensor_testing, scalar_add)
{
    tensor r(1, 1, 1, 3, {1, 2, 3});

    tensor l(1, 1, 1, 3, {3, 2, 1});

    tensor res(1, 1, 1, 3, {4, 4, 4});

    EXPECT_EQ(r + l, res);

    tensor r2(1, 1, 1, 3, {1, 1, 1});

    tensor l2(1, 1, 3, 1, {1, 1, 1});

    EXPECT_ANY_THROW(l2 + r2);
}

TEST(tensor_testing, scalar_mul)
{
    tensor r(1, 1, 3, 1, {1, 2, 3});

    tensor res(1, 1, 3, 1, {2, 4, 6});

    EXPECT_EQ(r * 2, res);
}

TEST(tensor_testing, mat_mul)
{
    tensor l1(1, 1, 1, 3, {1, 2, 3});

    tensor r1(1, 1, 3, 1, {4, 5, 6});

    tensor res(1, 1, 1, 1, {32});

    EXPECT_EQ(l1 * r1, res);

    tensor l2(1, 1, 1, 3, {1, 2, 3});

    tensor r2(1, 1, 1, 3, {1, 2, 3});

    EXPECT_ANY_THROW(l2 * r2);

    tensor l3(1, 1, 3, 2, {1, 2, 3, 4, 5, 6});

    tensor r3(1, 1, 2, 3, {7, 8, 9, 10, 11, 12});

    tensor res2(1, 1, 3, 3, {27, 30, 33, 61, 68, 75, 95, 106, 117});

    EXPECT_EQ(l3 * r3, res2);
}

TEST(operation_testing, bin_operation_test)
{
    tensor t1(1, 1, 1, 3, {1, 2, 3});

    tensor t2(1, 1, 1, 3, {3, 2, 1});

    std::shared_ptr<node_interface> input = std::make_shared<input_data>(t1);
    std::shared_ptr<scalar_add_op>  op    = std::make_shared<scalar_add_op>(input, t2);

    tensor res(1, 1, 1, 3, {4, 4, 4});

    EXPECT_EQ(op->evaluate(), res);
}

TEST(operation_testing, convol_test)
{
    tensor t1(1, 1, 3, 3, {1, 2, 3, 4, 5, 6, 7, 8, 9});

    tensor t2(1, 1, 2, 2, {-1, 0, 1, 2});

    std::shared_ptr<input_data> input = std::make_shared<input_data>(t1);
    std::shared_ptr<mat_convol_op> op = std::make_shared<mat_convol_op>(input, t2);

    tensor res(1, 1, 2, 2, {13, 15, 19, 21});

    EXPECT_EQ(op->evaluate(), res);
}

TEST(operation_testing, relu_test)
{
    tensor t1(1, 1, 3, 2, {1, 2, -1, 5, -6, -9});

    std::shared_ptr<input_data> input = std::make_shared<input_data>(t1);
    std::shared_ptr<relu_op> op       = std::make_shared<relu_op>(input);

    tensor t2(1, 1, 3, 2, {1, 2, 0, 5, 0, 0});

    EXPECT_EQ(op->evaluate(), t2);
}

TEST(operation_testing, softmax_test)
{
    tensor t1(1, 2, 2, 2, {2, 5, 3, 2, 3, 66, 7, 1});

    tensor res(1, 2, 2, 2, {0.2689, 0, 0.018, 0.7311, 0.7311, 1, 0.982, 0.2689});

    std::shared_ptr<input_data>input = std::make_shared<input_data>(t1);
    std::shared_ptr<softmax_op>op    = std::make_shared<softmax_op>(input);

    EXPECT_NO_THROW(op->evaluate());

    std::cout << "these values must be equal: " << op->evaluate() << " and " << res << std::endl;
}

TEST(neural_network_testing, simple_network)
{
    tensor in(2, 1, 3, 3, {6, 3, 2, 6, 7, 1, -6, 1, 6, 2, 1, 6, 8, 3, 1, 5, 80, 12});

    std::shared_ptr<input_data>input = std::make_shared<input_data>(in);

    tensor weight(2, 1, 3, 3, {1, 5, 3, 6, 8, 9, 2, 3, 5, 7, 2, 1, 5, 6, 7, 8, 3, 111});

    neural_network nn;
    nn.add_op(std::make_shared<matrix_mul_op>(input, weight));

    EXPECT_NO_THROW(nn.infer());
}

TEST(neural_network_testing, big_network)
{
    tensor in2(1, 1, 3, 2, {1, 2, 61, 132, 1, 5});

    number_t w1 = 1.2;

    tensor w2(1, 1, 2, 3, {5, 1, 6, 1, 2, 6});

    tensor w3(1, 1, 2, 2, {1, 2, 3, 4});

    tensor w4(1, 1, 1, 1, {4});
    
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

// TEST(optimization_testing, transpose_test)
// {
//     tensor t(1, 2, 2, 2, {1, 2, 3, 4, 5, 6, 7, 8});

//     tensor t_transposed(1, 2, 2, 2, {1, 3, 2, 4, 5, 7, 6, 8});

//     EXPECT_EQ(t.transpose(), t_transposed);
// }

#endif