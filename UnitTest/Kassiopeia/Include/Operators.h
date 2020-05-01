#ifndef UNIT_TEST_OPERATORS_H
#define UNIT_TEST_OPERATORS_H

#include "UnitTest.h"

#include "gtest/gtest.h"

//using namespace Kassiopeia;


/* Use fixture to set up tracks used in several test cases */
class KassiopeiaOperatorsTest : public TimeoutTest
{
  protected:
    void SetUp() override
    {
        TimeoutTest::SetUp();
    }

    void TearDown() override
    {
        TimeoutTest::TearDown();
    }
};

#endif /* UNIT_TEST_OPERATORS_H */
