#ifndef UNIT_TEST_OPERATORS_H
#define UNIT_TEST_OPERATORS_H

#include "UnitTest.h"

#include "UnitTest.h"

//using namespace Kassiopeia;


/* Use fixture to set up tracks used in several test cases */
class KassiopeiaOperatorsTest : public TimeoutTest
{
  public:
        KassiopeiaOperatorsTest()
    {
        TimeoutTest::SetUp();
    }

        ~KassiopeiaOperatorsTest()
    {
        TimeoutTest::TearDown();
    }
};

#endif /* UNIT_TEST_OPERATORS_H */
