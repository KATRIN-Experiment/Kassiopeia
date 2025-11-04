#ifndef UNIT_TEST_SHAPES_H
#define UNIT_TEST_SHAPES_H

#include "UnitTest.h"

//using namespace KGeoBag;


/* Use fixture to set up tracks used in several test cases */
class KGeoBagShapeTest : public TimeoutTest
{
  protected:
    void SetUp()
    {
        TimeoutTest::SetUp();
    }

    void TearDown()
    {
        TimeoutTest::TearDown();
    }
};

#endif /* UNIT_TEST_SHAPES_H */
