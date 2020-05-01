#ifndef UNIT_TEST_KEMFIELDTEST_HH
#define UNIT_TEST_KEMFIELDTEST_HH

#include "UnitTest.h"

#include "gtest/gtest.h"

//using namespace KEMField;


/* Use fixture to set up tracks used in several test cases */
class KEMFieldTest : public TimeoutTest
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

#endif /* UNIT_TEST_KEMFIELDTEST_HH */
