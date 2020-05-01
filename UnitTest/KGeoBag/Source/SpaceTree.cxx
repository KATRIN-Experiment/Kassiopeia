/**
 * Unit testing for very basic KGeoBag SpaceTree classes.
 * @author marco.kleesiek@kit.edu
 */

#include "KGPoint.hh"
#include "UnitTest.h"


using namespace KGeoBag;


/////////////////////////////////////////////////////////////////////////////
// SpaceTree Unit Testing
/////////////////////////////////////////////////////////////////////////////

TEST(KGeoBagSpaceTreeTest, KGPoint)
{
    KGPoint<3> p;

    EXPECT_EQ(0.0, p[0]);
    EXPECT_EQ(0.0, p[1]);
    EXPECT_EQ(0.0, p[2]);

    EXPECT_EQ(0.0, p.Magnitude());

    const double a[] = {1.0, 2.0, 3.0, 99999.9};
    p = a;

    EXPECT_NEAR(14.0, p.MagnitudeSquared(), 1E-5);
}
