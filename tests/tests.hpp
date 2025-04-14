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

#endif