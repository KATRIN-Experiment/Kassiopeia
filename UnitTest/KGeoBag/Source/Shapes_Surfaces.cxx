/**
 * Unit testing for KGeoBag shape classes
 * @author J. Behrens, N. Trost
 *
 * This file contains a unit tests for most of KGeoBag's shape classes.
 * All tests should be grouped together in a meaningful way, and should use
 * a fixture class which is derived from TimeoutTest. Please keep unit tests
 * as simple as possible (think of a minimal example of how to use a certain
 * class). It is a good idea to also include a death-test for each class.
 *
 * See the official GoogleTest pages for more info:
 *   https://code.google.com/p/googletest/wiki/Primer
 *   https://code.google.com/p/googletest/wiki/AdvancedGuide
 */

#include "KGAnnulusSurface.hh"
#include "KGConeSurface.hh"
#include "Shapes.h"

using namespace KGeoBag;


/////////////////////////////////////////////////////////////////////////////
// Shapes (Surfaces) Unit Testing
/////////////////////////////////////////////////////////////////////////////

TEST_F(KGeoBagShapeTest, KGAnnulusSurface)
{
    auto* tSurface = new KGAnnulusSurface();

    tSurface->Z(2.);
    tSurface->R1(0.5);
    tSurface->R2(1.);

    tSurface->AreaInitialize();

    EXPECT_EQ(tSurface->Z(), 2.);
    EXPECT_EQ(tSurface->R1(), 0.5);
    EXPECT_EQ(tSurface->R2(), 1.);

    delete tSurface;
}

TEST_F(KGeoBagShapeTest, KGAnnulusSurface_AreaAbove)
{
    auto* tSurface = new KGAnnulusSurface();

    tSurface->Z(2.);
    tSurface->R1(0.5);
    tSurface->R2(1.);

    tSurface->AreaInitialize();

    EXPECT_TRUE(tSurface->AreaAbove(KThreeVector(0.5, 0.5, 3.)));
    EXPECT_TRUE(tSurface->AreaAbove(KThreeVector(-0.5, -0.5, 3.)));
    EXPECT_TRUE(tSurface->AreaAbove(KThreeVector(-0.5, 0.5, 3.)));
    EXPECT_TRUE(tSurface->AreaAbove(KThreeVector(0.5, -0.5, 3.)));
    EXPECT_TRUE(tSurface->AreaAbove(KThreeVector(0.7, 0., 3.)));
    EXPECT_TRUE(tSurface->AreaAbove(KThreeVector(0., 0.7, 3.)));
    EXPECT_TRUE(tSurface->AreaAbove(KThreeVector(-0.7, 0., 3.)));
    EXPECT_TRUE(tSurface->AreaAbove(KThreeVector(0., -0.7, 3.)));

    EXPECT_TRUE(tSurface->AreaAbove(KThreeVector(1., 1., 3.)));
    EXPECT_TRUE(tSurface->AreaAbove(KThreeVector(-1., -1., 3.)));
    EXPECT_TRUE(tSurface->AreaAbove(KThreeVector(-1., 1., 3.)));
    EXPECT_TRUE(tSurface->AreaAbove(KThreeVector(1., -1., 3.)));
    EXPECT_TRUE(tSurface->AreaAbove(KThreeVector(1.4, 0., 3.)));
    EXPECT_TRUE(tSurface->AreaAbove(KThreeVector(0., 1.4, 3.)));
    EXPECT_TRUE(tSurface->AreaAbove(KThreeVector(-1.4, 0., 3.)));
    EXPECT_TRUE(tSurface->AreaAbove(KThreeVector(0., -1.4, 3.)));

    EXPECT_FALSE(tSurface->AreaAbove(KThreeVector(0.5, 0.5, 1.)));
    EXPECT_FALSE(tSurface->AreaAbove(KThreeVector(-0.5, -0.5, 1.)));
    EXPECT_FALSE(tSurface->AreaAbove(KThreeVector(-0.5, 0.5, 1.)));
    EXPECT_FALSE(tSurface->AreaAbove(KThreeVector(0.5, -0.5, 1.)));
    EXPECT_FALSE(tSurface->AreaAbove(KThreeVector(0.7, 0., 1.)));
    EXPECT_FALSE(tSurface->AreaAbove(KThreeVector(0., 0.7, 1.)));
    EXPECT_FALSE(tSurface->AreaAbove(KThreeVector(-0.7, 0., 1.)));
    EXPECT_FALSE(tSurface->AreaAbove(KThreeVector(0., -0.7, 1.)));

    EXPECT_FALSE(tSurface->AreaAbove(KThreeVector(0.5, 0.5, -1.)));
    EXPECT_FALSE(tSurface->AreaAbove(KThreeVector(-0.5, -0.5, -1.)));
    EXPECT_FALSE(tSurface->AreaAbove(KThreeVector(-0.5, 0.5, -1.)));
    EXPECT_FALSE(tSurface->AreaAbove(KThreeVector(0.5, -0.5, -1.)));
    EXPECT_FALSE(tSurface->AreaAbove(KThreeVector(0.7, 0., -1.)));
    EXPECT_FALSE(tSurface->AreaAbove(KThreeVector(0., 0.7, -1.)));
    EXPECT_FALSE(tSurface->AreaAbove(KThreeVector(-0.7, 0., -1.)));
    EXPECT_FALSE(tSurface->AreaAbove(KThreeVector(0., -0.7, -1.)));

    delete tSurface;

    tSurface = new KGAnnulusSurface();

    tSurface->Z(0.);
    tSurface->R1(0.5);
    tSurface->R2(1.);

    tSurface->AreaInitialize();

    EXPECT_TRUE(tSurface->AreaAbove(KThreeVector(0.5, 0.5, 1.)));
    EXPECT_TRUE(tSurface->AreaAbove(KThreeVector(-0.5, -0.5, 1.)));
    EXPECT_TRUE(tSurface->AreaAbove(KThreeVector(-0.5, 0.5, 1.)));
    EXPECT_TRUE(tSurface->AreaAbove(KThreeVector(0.5, -0.5, 1.)));
    EXPECT_TRUE(tSurface->AreaAbove(KThreeVector(0.7, 0., 1.)));
    EXPECT_TRUE(tSurface->AreaAbove(KThreeVector(0., 0.7, 1.)));
    EXPECT_TRUE(tSurface->AreaAbove(KThreeVector(-0.7, 0., 1.)));
    EXPECT_TRUE(tSurface->AreaAbove(KThreeVector(0., -0.7, 1.)));

    EXPECT_TRUE(tSurface->AreaAbove(KThreeVector(1., 1., 1.)));
    EXPECT_TRUE(tSurface->AreaAbove(KThreeVector(-1., -1., 1.)));
    EXPECT_TRUE(tSurface->AreaAbove(KThreeVector(-1., 1., 1.)));
    EXPECT_TRUE(tSurface->AreaAbove(KThreeVector(1., -1., 1.)));
    EXPECT_TRUE(tSurface->AreaAbove(KThreeVector(1.4, 0., 1.)));
    EXPECT_TRUE(tSurface->AreaAbove(KThreeVector(0., 1.4, 1.)));
    EXPECT_TRUE(tSurface->AreaAbove(KThreeVector(-1.4, 0., 1.)));
    EXPECT_TRUE(tSurface->AreaAbove(KThreeVector(0., -1.4, 1.)));

    EXPECT_FALSE(tSurface->AreaAbove(KThreeVector(0.5, 0.5, -1.)));
    EXPECT_FALSE(tSurface->AreaAbove(KThreeVector(-0.5, -0.5, -1.)));
    EXPECT_FALSE(tSurface->AreaAbove(KThreeVector(-0.5, 0.5, -1.)));
    EXPECT_FALSE(tSurface->AreaAbove(KThreeVector(0.5, -0.5, -1.)));
    EXPECT_FALSE(tSurface->AreaAbove(KThreeVector(0.7, 0., -1.)));
    EXPECT_FALSE(tSurface->AreaAbove(KThreeVector(0., 0.7, -1.)));
    EXPECT_FALSE(tSurface->AreaAbove(KThreeVector(-0.7, 0., -1.)));
    EXPECT_FALSE(tSurface->AreaAbove(KThreeVector(0., -0.7, -1.)));

    delete tSurface;
}

TEST_F(KGeoBagShapeTest, KGAnnulusSurface_AreaNormal)
{
    auto* tSurface = new KGAnnulusSurface();

    tSurface->Z(2.);
    tSurface->R1(0.5);
    tSurface->R2(1.);

    tSurface->AreaInitialize();

    EXPECT_EQ(tSurface->AreaNormal(KThreeVector(0.5, 0.5, 3.)), KThreeVector(0., 0., 1.));
    EXPECT_EQ(tSurface->AreaNormal(KThreeVector(-0.5, -0.5, 3.)), KThreeVector(0., 0., 1.));
    EXPECT_EQ(tSurface->AreaNormal(KThreeVector(0.5, 0.5, 3.)), KThreeVector(0., 0., 1.));
    EXPECT_EQ(tSurface->AreaNormal(KThreeVector(0.5, -0.5, 3.)), KThreeVector(0., 0., 1.));
    EXPECT_EQ(tSurface->AreaNormal(KThreeVector(0.7, 0.0, 3.)), KThreeVector(0., 0., 1.));
    EXPECT_EQ(tSurface->AreaNormal(KThreeVector(0.0, 0.7, 3.)), KThreeVector(0., 0., 1.));
    EXPECT_EQ(tSurface->AreaNormal(KThreeVector(-0.7, 0.0, 3.)), KThreeVector(0., 0., 1.));
    EXPECT_EQ(tSurface->AreaNormal(KThreeVector(0.0, -0.7, 3.)), KThreeVector(0., 0., 1.));

    EXPECT_EQ(tSurface->AreaNormal(KThreeVector(0.6, 0.6, 1.)), KThreeVector(0., 0., 1.));
    EXPECT_EQ(tSurface->AreaNormal(KThreeVector(-0.6, -0.6, 1.)), KThreeVector(0., 0., 1.));
    EXPECT_EQ(tSurface->AreaNormal(KThreeVector(0.6, 0.6, 1.)), KThreeVector(0., 0., 1.));
    EXPECT_EQ(tSurface->AreaNormal(KThreeVector(0.6, -0.6, 1.)), KThreeVector(0., 0., 1.));
    EXPECT_EQ(tSurface->AreaNormal(KThreeVector(0.7, 0.0, 1.)), KThreeVector(0., 0., 1.));
    EXPECT_EQ(tSurface->AreaNormal(KThreeVector(0.0, 0.7, 1.)), KThreeVector(0., 0., 1.));
    EXPECT_EQ(tSurface->AreaNormal(KThreeVector(-0.7, 0.0, 1.)), KThreeVector(0., 0., 1.));
    EXPECT_EQ(tSurface->AreaNormal(KThreeVector(0.0, -0.7, 1.)), KThreeVector(0., 0., 1.));

    delete tSurface;
}

TEST_F(KGeoBagShapeTest, KGAnnulusSurface_AbovePoint)
{
    auto* tSurface = new KGAnnulusSurface();

    tSurface->Z(2.);
    tSurface->R1(0.5);
    tSurface->R2(1.);

    tSurface->AreaInitialize();

    EXPECT_VECTOR_NEAR(tSurface->AreaPoint(KThreeVector(0.5, 0.5, 3.)), KThreeVector(0.5, 0.5, 2.));
    EXPECT_VECTOR_NEAR(tSurface->AreaPoint(KThreeVector(0.5, 0.5, 1.)), KThreeVector(0.5, 0.5, 2.));

    delete tSurface;
}

//////////////////////////////////////////////////////////////////////////////

TEST_F(KGeoBagShapeTest, KGConeSurface)
{
    auto* tSurface = new KGConeSurface();

    tSurface->ZA(2.);
    tSurface->ZB(0.);
    tSurface->RB(1.);

    tSurface->AreaInitialize();

    EXPECT_EQ(tSurface->ZA(), 2.);
    EXPECT_EQ(tSurface->ZB(), 0.);
    EXPECT_EQ(tSurface->RB(), 1.);

    delete tSurface;
}

TEST_F(KGeoBagShapeTest, KGConeSurface_AreaAbove)
{
    auto* tSurface = new KGConeSurface();

    tSurface->ZA(2.);
    tSurface->ZB(0.);
    tSurface->RB(1.);

    tSurface->AreaInitialize();

    EXPECT_TRUE(tSurface->AreaAbove(KThreeVector(0.55, 0., 1.)));
    EXPECT_TRUE(tSurface->AreaAbove(KThreeVector(0., 0.55, 1.)));
    EXPECT_TRUE(tSurface->AreaAbove(KThreeVector(0., 0., 3.)));
    EXPECT_FALSE(tSurface->AreaAbove(KThreeVector(0., 0., 1.)));

    delete tSurface;
}

TEST_F(KGeoBagShapeTest, KGConeSurface_AreaNormal)
{
    auto* tSurface = new KGConeSurface();

    tSurface->ZA(3.);
    tSurface->ZB(0.);
    tSurface->RB(1.);

    tSurface->AreaInitialize();

    EXPECT_VECTOR_NEAR(tSurface->AreaNormal(KThreeVector(0., 11. / 3., 2.)), KThreeVector(0., 3., 1.).Unit());

    delete tSurface;
}

TEST_F(KGeoBagShapeTest, KGConeSurface_PointAbove)
{
    auto* tSurface = new KGConeSurface();

    tSurface->ZA(3.);
    tSurface->ZB(0.);
    tSurface->RB(1.);

    tSurface->AreaInitialize();

    EXPECT_VECTOR_NEAR(tSurface->AreaPoint(KThreeVector(0., 11. / 3., 2.)), KThreeVector(0., 2. / 3., 1.));

    delete tSurface;
}

//////////////////////////////////////////////////////////////////////////////
// DEATH TESTS ///////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

TEST(KGeoBagShapeDeathTest, KGAnnulusSurface) {}

TEST(KGeoBagShapeDeathTest, KGConeSurface) {}
