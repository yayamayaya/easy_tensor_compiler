#include "tests.hpp"
#include "logging.hpp"

int main()
{
    logger::logger_init("gtest.log");

    testing::InitGoogleTest();

    return RUN_ALL_TESTS();
}
