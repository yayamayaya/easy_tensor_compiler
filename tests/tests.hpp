#ifndef TESTS_H
#define TESTS_H

#include <gtest/gtest.h>
#include "tensor_compiler.hpp"

TEST(tensor_testing, tensor_init)
{
    tensor t(1, 1, 2, 3);

    t = {1, 2, 3, 3, 2, 1};

    EXPECT_NO_THROW(t(0, 0, 1, 2));
}

TEST(tensor_testing, tensor_print)
{
    tensor t(1, 1, 2, 2);

    t = std::vector<number_t>{1, 2, 3, 4};

    std::cout << t << std::endl;
}

TEST(tensor_testing, out_of_range_ind)
{
    tensor t(1, 1, 2, 2);

    EXPECT_ANY_THROW(t(2, 2, 2, 2));
}

TEST(math_testing, scalar_adds_op)
{
    tensor t1(1, 1, 1, 2);
    tensor t2(1, 1, 1, 2);

    t1 = {5, 6};
    t2 = {3, 5};

    _LOG << "first tensor:  " << t1 << END_;
    _LOG << "second tensor: " << t2 << END_;

    _LOG << "sum result: " << t1 + t2 << END_;

    // const auto &input = std::make_shared<inе>
}

#endif