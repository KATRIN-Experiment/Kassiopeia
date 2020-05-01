/**
 * Unit testing for Kassiopeia terminator classes
 * @author J. Behrens, N. Trost
 *
 * This file contains a unit tests for most of Kassiopeia's terminator classes.
 * All tests should be grouped together in a meaningful way, and should use
 * a fixture class which is derived from TimeoutTest. Please keep unit tests
 * as simple as possible (think of a minimal example of how to use a certain
 * class). It is a good idea to also include a death-test for each class.
 *
 * See the official GoogleTest pages for more info:
 *   https://code.google.com/p/googletest/wiki/Primer
 *   https://code.google.com/p/googletest/wiki/AdvancedGuide
 */

#include "Terminators.h"

#include "KSTermDeath.h"
#include "KSTermMaxEnergy.h"
#include "KSTermMaxLength.h"
#include "KSTermMaxLongEnergy.h"
#include "KSTermMaxR.h"
#include "KSTermMaxSteps.h"
#include "KSTermMaxTime.h"
#include "KSTermMaxZ.h"
#include "KSTermMinEnergy.h"
#include "KSTermMinLongEnergy.h"
#include "KSTermMinR.h"
#include "KSTermMinZ.h"
#include "KSTermTrapped.h"

using namespace Kassiopeia;
using namespace std;


//////////////////////////////////////////////////////////////////////////////
// Terminators Unit Testing
//////////////////////////////////////////////////////////////////////////////

TEST_F(KassiopeiaTerminatorTest, KSTermDeath)
{
    ASSERT_EQ(fParticles->size(), 0U);

    auto* tTerminator = new KSTermDeath();
    ASSERT_PTR(tTerminator);

    std::string tName = "test_label";

    tTerminator->SetName(tName);
    EXPECT_STRING_EQ(tTerminator->GetName(), tName);

    bool tResult = false;
    tTerminator->CalculateTermination(*fInitialParticle, tResult);
    EXPECT_TRUE(tResult);

    tTerminator->ExecuteTermination(*fInitialParticle, *fFinalParticle, *fParticles);
    EXPECT_FALSE(fFinalParticle->IsActive());
    fFinalParticle->ReleaseLabel(tName);

    delete tTerminator;
}

//////////////////////////////////////////////////////////////////////////////

TEST_F(KassiopeiaTerminatorTest, KSTermMaxEnergy)
{
    ASSERT_EQ(fParticles->size(), 0U);

    auto* tTerminator = new KSTermMaxEnergy();
    ASSERT_PTR(tTerminator);

    std::string tName = "test_label";
    tTerminator->SetName(tName);
    EXPECT_STRING_EQ(tTerminator->GetName(), tName);

    bool tResult = true;
    tTerminator->CalculateTermination(*fInitialParticle, tResult);
    EXPECT_FALSE(tResult);

    tTerminator->ExecuteTermination(*fInitialParticle, *fFinalParticle, *fParticles);
    EXPECT_FALSE(fFinalParticle->IsActive());
    fFinalParticle->ReleaseLabel(tName);

    tTerminator->SetMaxEnergy(100.123);

    tResult = true;
    fInitialParticle->SetKineticEnergy_eV(100.);
    tTerminator->CalculateTermination(*fInitialParticle, tResult);
    EXPECT_FALSE(tResult);

    tResult = true;
    fInitialParticle->SetKineticEnergy_eV(0.);
    tTerminator->CalculateTermination(*fInitialParticle, tResult);
    EXPECT_FALSE(tResult);

    tResult = false;
    fInitialParticle->SetKineticEnergy_eV(101.);
    tTerminator->CalculateTermination(*fInitialParticle, tResult);
    EXPECT_TRUE(tResult);

    tTerminator->SetMaxEnergy(0.);

    tResult = false;
    fInitialParticle->SetKineticEnergy_eV(EPSILON_DOUBLE);
    tTerminator->CalculateTermination(*fInitialParticle, tResult);
    EXPECT_TRUE(tResult);

    delete tTerminator;
}

//////////////////////////////////////////////////////////////////////////////

TEST_F(KassiopeiaTerminatorTest, KSTermMaxLength)
{
    ASSERT_EQ(fParticles->size(), 0U);

    auto* tTerminator = new KSTermMaxLength();
    ASSERT_PTR(tTerminator);

    std::string tName = "test_label";
    tTerminator->SetName(tName);
    EXPECT_STRING_EQ(tTerminator->GetName(), tName);

    bool tResult = true;
    tTerminator->CalculateTermination(*fInitialParticle, tResult);
    EXPECT_FALSE(tResult);

    tTerminator->ExecuteTermination(*fInitialParticle, *fFinalParticle, *fParticles);
    EXPECT_FALSE(fFinalParticle->IsActive());
    fFinalParticle->ReleaseLabel(tName);

    tTerminator->SetLength(100.123);

    tResult = true;
    fInitialParticle->SetLength(100.);
    tTerminator->CalculateTermination(*fInitialParticle, tResult);
    EXPECT_FALSE(tResult);

    tResult = true;
    fInitialParticle->SetLength(0.);
    tTerminator->CalculateTermination(*fInitialParticle, tResult);
    EXPECT_FALSE(tResult);

    tResult = false;
    fInitialParticle->SetLength(101.);
    tTerminator->CalculateTermination(*fInitialParticle, tResult);
    EXPECT_TRUE(tResult);

    tTerminator->SetLength(0.);

    tResult = false;
    fInitialParticle->SetLength(EPSILON_DOUBLE);
    tTerminator->CalculateTermination(*fInitialParticle, tResult);
    EXPECT_TRUE(tResult);

    delete tTerminator;
}

//////////////////////////////////////////////////////////////////////////////

TEST_F(KassiopeiaTerminatorTest, KSTermMaxLongEnergy)
{
    ASSERT_EQ(fParticles->size(), 0U);

    auto* tTerminator = new KSTermMaxLongEnergy();
    ASSERT_PTR(tTerminator);

    std::string tName = "test_label";
    tTerminator->SetName(tName);
    EXPECT_STRING_EQ(tTerminator->GetName(), tName);

    bool tResult = true;
    tTerminator->CalculateTermination(*fInitialParticle, tResult);
    EXPECT_FALSE(tResult);

    tTerminator->ExecuteTermination(*fInitialParticle, *fFinalParticle, *fParticles);
    EXPECT_FALSE(fFinalParticle->IsActive());
    fFinalParticle->ReleaseLabel(tName);

    tTerminator->SetMaxLongEnergy(100.123);

    tResult = true;
    fInitialParticle->SetKineticEnergy_eV(100.);
    fInitialParticle->SetPolarAngleToB(0.);
    tTerminator->CalculateTermination(*fInitialParticle, tResult);
    EXPECT_FALSE(tResult);

    tResult = true;
    fInitialParticle->SetKineticEnergy_eV(0.);
    fInitialParticle->SetPolarAngleToB(0.);
    tTerminator->CalculateTermination(*fInitialParticle, tResult);
    EXPECT_FALSE(tResult);

    tResult = false;
    fInitialParticle->SetKineticEnergy_eV(101.);
    fInitialParticle->SetPolarAngleToB(0.);
    tTerminator->CalculateTermination(*fInitialParticle, tResult);
    EXPECT_TRUE(tResult);

    tResult = true;
    fInitialParticle->SetKineticEnergy_eV(145.);
    fInitialParticle->SetPolarAngleToB(90.);
    tTerminator->CalculateTermination(*fInitialParticle, tResult);
    EXPECT_FALSE(tResult);

    tTerminator->SetMaxLongEnergy(0.123);

    tResult = false;
    fInitialParticle->SetKineticEnergy_eV(101.);
    fInitialParticle->SetPolarAngleToB(90.);
    tTerminator->CalculateTermination(*fInitialParticle, tResult);
    EXPECT_FALSE(tResult);

    tTerminator->SetMaxLongEnergy(0.);

    tResult = false;
    fInitialParticle->SetKineticEnergy_eV(EPSILON_DOUBLE);
    fInitialParticle->SetPolarAngleToB(EPSILON_DOUBLE);
    tTerminator->CalculateTermination(*fInitialParticle, tResult);
    EXPECT_TRUE(tResult);

    delete tTerminator;
}

//////////////////////////////////////////////////////////////////////////////

TEST_F(KassiopeiaTerminatorTest, KSTermMaxR)
{
    ASSERT_EQ(fParticles->size(), 0U);

    KThreeVector tRadialVector = KThreeVector(1., 0., 0.);

    auto* tTerminator = new KSTermMaxR();
    ASSERT_PTR(tTerminator);

    std::string tName = "test_label";
    tTerminator->SetName(tName);
    EXPECT_STRING_EQ(tTerminator->GetName(), tName);

    bool tResult = true;
    tTerminator->CalculateTermination(*fInitialParticle, tResult);
    EXPECT_FALSE(tResult);

    tTerminator->ExecuteTermination(*fInitialParticle, *fFinalParticle, *fParticles);
    EXPECT_FALSE(fFinalParticle->IsActive());
    fFinalParticle->ReleaseLabel(tName);

    tTerminator->SetMaxR(100.123);

    tResult = true;
    fInitialParticle->SetPosition(99. * tRadialVector);
    tTerminator->CalculateTermination(*fInitialParticle, tResult);
    EXPECT_FALSE(tResult);

    tResult = true;
    fInitialParticle->SetPosition(0. * tRadialVector);
    tTerminator->CalculateTermination(*fInitialParticle, tResult);
    EXPECT_FALSE(tResult);

    tResult = false;
    fInitialParticle->SetPosition(101. * tRadialVector);
    tTerminator->CalculateTermination(*fInitialParticle, tResult);
    EXPECT_TRUE(tResult);

    tTerminator->SetMaxR(0.);

    tResult = false;
    fInitialParticle->SetPosition(EPSILON_DOUBLE * tRadialVector);
    tTerminator->CalculateTermination(*fInitialParticle, tResult);
    EXPECT_TRUE(tResult);

    delete tTerminator;
}

//////////////////////////////////////////////////////////////////////////////

TEST_F(KassiopeiaTerminatorTest, KSTermMaxSteps)
{
    ASSERT_EQ(fParticles->size(), 0U);

    auto* tTerminator = new KSTermMaxSteps();
    ASSERT_PTR(tTerminator);

    std::string tName = "test_label";
    tTerminator->SetName(tName);
    EXPECT_STRING_EQ(tTerminator->GetName(), tName);

    bool tResult = false;
    tTerminator->CalculateTermination(*fInitialParticle, tResult);
    EXPECT_TRUE(tResult);  // should terminate because "fSteps >= fMaxSteps"

    tTerminator->ExecuteTermination(*fInitialParticle, *fFinalParticle, *fParticles);
    EXPECT_FALSE(fFinalParticle->IsActive());
    fFinalParticle->ReleaseLabel(tName);

    tTerminator->SetMaxSteps(0);

    tResult = false;
    tTerminator->CalculateTermination(*fInitialParticle, tResult);
    EXPECT_TRUE(tResult);

    tTerminator->SetMaxSteps(100);

    for (unsigned int tSteps = 1; tSteps <= 100; tSteps++)  // start at step 1 (internal step count is 1 here)
    {
        tResult = true;
        tTerminator->CalculateTermination(*fInitialParticle, tResult);
        EXPECT_FALSE(tResult);
    }

    tResult = false;
    tTerminator->CalculateTermination(*fInitialParticle, tResult);
    EXPECT_TRUE(tResult);

    delete tTerminator;
}

//////////////////////////////////////////////////////////////////////////////

TEST_F(KassiopeiaTerminatorTest, KSTermMaxTime)
{
    ASSERT_EQ(fParticles->size(), 0U);

    auto* tTerminator = new KSTermMaxTime();
    ASSERT_PTR(tTerminator);

    std::string tName = "test_label";
    tTerminator->SetName(tName);
    EXPECT_STRING_EQ(tTerminator->GetName(), tName);

    bool tResult = true;
    tTerminator->CalculateTermination(*fInitialParticle, tResult);
    EXPECT_FALSE(tResult);

    tTerminator->ExecuteTermination(*fInitialParticle, *fFinalParticle, *fParticles);
    EXPECT_FALSE(fFinalParticle->IsActive());
    fFinalParticle->ReleaseLabel(tName);

    tTerminator->SetTime(100.123);

    tResult = true;
    fInitialParticle->SetTime(99.);
    tTerminator->CalculateTermination(*fInitialParticle, tResult);
    EXPECT_FALSE(tResult);

    tResult = true;
    fInitialParticle->SetTime(0.);
    tTerminator->CalculateTermination(*fInitialParticle, tResult);
    EXPECT_FALSE(tResult);

    tResult = false;
    fInitialParticle->SetTime(101.);
    tTerminator->CalculateTermination(*fInitialParticle, tResult);
    EXPECT_TRUE(tResult);

    tTerminator->SetTime(0.);

    tResult = false;
    fInitialParticle->SetTime(EPSILON_DOUBLE);
    tTerminator->CalculateTermination(*fInitialParticle, tResult);
    EXPECT_TRUE(tResult);

    delete tTerminator;
}

//////////////////////////////////////////////////////////////////////////////

TEST_F(KassiopeiaTerminatorTest, KSTermMaxZ)
{
    ASSERT_EQ(fParticles->size(), 0U);

    KThreeVector tAxialVector = KThreeVector(0., 0., 1.);

    auto* tTerminator = new KSTermMaxZ();
    ASSERT_PTR(tTerminator);

    std::string tName = "test_label";
    tTerminator->SetName(tName);
    EXPECT_STRING_EQ(tTerminator->GetName(), tName);

    bool tResult = true;
    tTerminator->CalculateTermination(*fInitialParticle, tResult);
    EXPECT_FALSE(tResult);

    tTerminator->ExecuteTermination(*fInitialParticle, *fFinalParticle, *fParticles);
    EXPECT_FALSE(fFinalParticle->IsActive());
    fFinalParticle->ReleaseLabel(tName);

    tTerminator->SetMaxZ(100.123);

    tResult = true;
    fInitialParticle->SetPosition(99. * tAxialVector);
    tTerminator->CalculateTermination(*fInitialParticle, tResult);
    EXPECT_FALSE(tResult);

    tResult = true;
    fInitialParticle->SetPosition(0. * tAxialVector);
    tTerminator->CalculateTermination(*fInitialParticle, tResult);
    EXPECT_FALSE(tResult);

    tResult = false;
    fInitialParticle->SetPosition(101. * tAxialVector);
    tTerminator->CalculateTermination(*fInitialParticle, tResult);
    EXPECT_TRUE(tResult);

    tTerminator->SetMaxZ(0.);

    tResult = false;
    fInitialParticle->SetPosition(EPSILON_DOUBLE * tAxialVector);
    tTerminator->CalculateTermination(*fInitialParticle, tResult);
    EXPECT_TRUE(tResult);

    delete tTerminator;
}

//////////////////////////////////////////////////////////////////////////////

TEST_F(KassiopeiaTerminatorTest, KSTermMinEnergy)
{
    ASSERT_EQ(fParticles->size(), 0U);

    auto* tTerminator = new KSTermMinEnergy();
    ASSERT_PTR(tTerminator);

    std::string tName = "test_label";
    tTerminator->SetName(tName);
    EXPECT_STRING_EQ(tTerminator->GetName(), tName);

    bool tResult = true;
    tTerminator->CalculateTermination(*fInitialParticle, tResult);
    EXPECT_FALSE(tResult);

    tTerminator->ExecuteTermination(*fInitialParticle, *fFinalParticle, *fParticles);
    EXPECT_FALSE(fFinalParticle->IsActive());
    fFinalParticle->ReleaseLabel(tName);

    tTerminator->SetMinEnergy(100.123);

    tResult = false;
    fInitialParticle->SetKineticEnergy_eV(100.);
    tTerminator->CalculateTermination(*fInitialParticle, tResult);
    EXPECT_TRUE(tResult);

    tResult = false;
    fInitialParticle->SetKineticEnergy_eV(0.);
    tTerminator->CalculateTermination(*fInitialParticle, tResult);
    EXPECT_TRUE(tResult);

    tResult = true;
    fInitialParticle->SetKineticEnergy_eV(101.);
    tTerminator->CalculateTermination(*fInitialParticle, tResult);
    EXPECT_FALSE(tResult);

    tTerminator->SetMinEnergy(0.);

    tResult = true;
    fInitialParticle->SetKineticEnergy_eV(EPSILON_DOUBLE);
    tTerminator->CalculateTermination(*fInitialParticle, tResult);
    EXPECT_FALSE(tResult);

    delete tTerminator;
}

//////////////////////////////////////////////////////////////////////////////

TEST_F(KassiopeiaTerminatorTest, KSTermMinLongEnergy)
{
    ASSERT_EQ(fParticles->size(), 0U);

    auto* tTerminator = new KSTermMinLongEnergy();
    ASSERT_PTR(tTerminator);

    std::string tName = "test_label";
    tTerminator->SetName(tName);
    EXPECT_STRING_EQ(tTerminator->GetName(), tName);

    bool tResult = true;
    tTerminator->CalculateTermination(*fInitialParticle, tResult);
    EXPECT_FALSE(tResult);

    tTerminator->ExecuteTermination(*fInitialParticle, *fFinalParticle, *fParticles);
    EXPECT_FALSE(fFinalParticle->IsActive());
    fFinalParticle->ReleaseLabel(tName);

    tTerminator->SetMinLongEnergy(100.123);

    tResult = false;
    fInitialParticle->SetKineticEnergy_eV(100.);
    fInitialParticle->SetPolarAngleToB(0.);
    tTerminator->CalculateTermination(*fInitialParticle, tResult);
    EXPECT_TRUE(tResult);

    tResult = false;
    fInitialParticle->SetKineticEnergy_eV(0.);
    fInitialParticle->SetPolarAngleToB(0.);
    tTerminator->CalculateTermination(*fInitialParticle, tResult);
    EXPECT_TRUE(tResult);

    tResult = true;
    fInitialParticle->SetKineticEnergy_eV(101.);
    fInitialParticle->SetPolarAngleToB(0.);
    tTerminator->CalculateTermination(*fInitialParticle, tResult);
    EXPECT_FALSE(tResult);

    tResult = false;
    fInitialParticle->SetKineticEnergy_eV(145.);
    fInitialParticle->SetPolarAngleToB(90.);
    tTerminator->CalculateTermination(*fInitialParticle, tResult);
    EXPECT_TRUE(tResult);

    tTerminator->SetMinLongEnergy(0.123);

    tResult = false;
    fInitialParticle->SetKineticEnergy_eV(101.);
    fInitialParticle->SetPolarAngleToB(90.);
    tTerminator->CalculateTermination(*fInitialParticle, tResult);
    EXPECT_TRUE(tResult);

    tTerminator->SetMinLongEnergy(0.);

    tResult = true;
    fInitialParticle->SetKineticEnergy_eV(EPSILON_DOUBLE);
    fInitialParticle->SetPolarAngleToB(EPSILON_DOUBLE);
    tTerminator->CalculateTermination(*fInitialParticle, tResult);
    EXPECT_FALSE(tResult);

    delete tTerminator;
}

//////////////////////////////////////////////////////////////////////////////

TEST_F(KassiopeiaTerminatorTest, KSTermMinR)
{
    ASSERT_EQ(fParticles->size(), 0U);

    KThreeVector tRadialVector = KThreeVector(1., 0., 0.);

    auto* tTerminator = new KSTermMinR();
    ASSERT_PTR(tTerminator);

    std::string tName = "test_label";
    tTerminator->SetName(tName);
    EXPECT_STRING_EQ(tTerminator->GetName(), tName);

    bool tResult = true;
    tTerminator->CalculateTermination(*fInitialParticle, tResult);
    EXPECT_FALSE(tResult);

    tTerminator->ExecuteTermination(*fInitialParticle, *fFinalParticle, *fParticles);
    EXPECT_FALSE(fFinalParticle->IsActive());
    fFinalParticle->ReleaseLabel(tName);

    tTerminator->SetMinR(100.123);

    tResult = false;
    fInitialParticle->SetPosition(99. * tRadialVector);
    tTerminator->CalculateTermination(*fInitialParticle, tResult);
    EXPECT_TRUE(tResult);

    tResult = false;
    fInitialParticle->SetPosition(0. * tRadialVector);
    tTerminator->CalculateTermination(*fInitialParticle, tResult);
    EXPECT_TRUE(tResult);

    tResult = true;
    fInitialParticle->SetPosition(101. * tRadialVector);
    tTerminator->CalculateTermination(*fInitialParticle, tResult);
    EXPECT_FALSE(tResult);

    tTerminator->SetMinR(0.);

    tResult = true;
    fInitialParticle->SetPosition(EPSILON_DOUBLE * tRadialVector);
    tTerminator->CalculateTermination(*fInitialParticle, tResult);
    EXPECT_FALSE(tResult);

    delete tTerminator;
}

//////////////////////////////////////////////////////////////////////////////

TEST_F(KassiopeiaTerminatorTest, KSTermMinZ)
{
    ASSERT_EQ(fParticles->size(), 0U);

    KThreeVector tAxialVector = KThreeVector(0., 0., 1.);

    auto* tTerminator = new KSTermMinZ();
    ASSERT_PTR(tTerminator);

    std::string tName = "test_label";
    tTerminator->SetName(tName);
    EXPECT_STRING_EQ(tTerminator->GetName(), tName);

    bool tResult = true;
    tTerminator->CalculateTermination(*fInitialParticle, tResult);
    EXPECT_FALSE(tResult);

    tTerminator->ExecuteTermination(*fInitialParticle, *fFinalParticle, *fParticles);
    EXPECT_FALSE(fFinalParticle->IsActive());
    fFinalParticle->ReleaseLabel(tName);

    tTerminator->SetMinZ(100.123);

    tResult = false;
    fInitialParticle->SetPosition(99. * tAxialVector);
    tTerminator->CalculateTermination(*fInitialParticle, tResult);
    EXPECT_TRUE(tResult);

    tResult = false;
    fInitialParticle->SetPosition(0. * tAxialVector);
    tTerminator->CalculateTermination(*fInitialParticle, tResult);
    EXPECT_TRUE(tResult);

    tResult = true;
    fInitialParticle->SetPosition(101. * tAxialVector);
    tTerminator->CalculateTermination(*fInitialParticle, tResult);
    EXPECT_FALSE(tResult);

    tTerminator->SetMinZ(0.);

    tResult = true;
    fInitialParticle->SetPosition(EPSILON_DOUBLE * tAxialVector);
    tTerminator->CalculateTermination(*fInitialParticle, tResult);
    EXPECT_FALSE(tResult);

    delete tTerminator;
}

//////////////////////////////////////////////////////////////////////////////

TEST_F(KassiopeiaTerminatorTest, KSTermTrapped)
{
    ASSERT_EQ(fParticles->size(), 0U);

    auto* tTerminator = new KSTermTrapped();
    ASSERT_PTR(tTerminator);

    std::string tName = "test_label";
    tTerminator->SetName(tName);
    EXPECT_STRING_EQ(tTerminator->GetName(), tName);

    bool tResult = true;
    tTerminator->CalculateTermination(*fInitialParticle, tResult);
    EXPECT_FALSE(tResult);

    tTerminator->ExecuteTermination(*fInitialParticle, *fFinalParticle, *fParticles);
    EXPECT_FALSE(fFinalParticle->IsActive());
    fFinalParticle->ReleaseLabel(tName);

    tTerminator->SetMaxTurns(0);

    tResult = true;
    tTerminator->CalculateTermination(*fInitialParticle, tResult);
    EXPECT_FALSE(tResult);  // should not terminate because no turn-around happened yet

    tTerminator->SetMaxTurns(100);

    KThreeVector tMomentum(0., 0., 1.);
    fInitialParticle->SetMagneticField(KThreeVector(0., 0., 1.));

    for (unsigned int tSteps = 0; tSteps < 100; tSteps++) {
        tResult = true;
        fInitialParticle->SetMomentum(tMomentum);
        tTerminator->CalculateTermination(*fInitialParticle, tResult);
        EXPECT_FALSE(tResult);
        tMomentum = -1. * tMomentum;  // turn around
    }

    tResult = false;
    fInitialParticle->SetMomentum(tMomentum);
    tTerminator->CalculateTermination(*fInitialParticle, tResult);
    EXPECT_TRUE(tResult);

    delete tTerminator;
}

//////////////////////////////////////////////////////////////////////////////
// DEATH TESTS ///////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

TEST(KassiopeiaTerminatorDeathTest, KSTermMaxEnergy)  // test case name should end in "DeathTest"
{
    KSParticle tInitialParticle;
    bool tResult = true;
    auto* tTerminator = new KSTermMaxEnergy();
    tTerminator->SetMaxEnergy(10.);
    tTerminator->SetMaxEnergy(-1.);
    ASSERT_ANY_THROW(tTerminator->CalculateTermination(tInitialParticle, tResult));
}

TEST(KassiopeiaTerminatorDeathTest, KSTermMaxLength)  // test case name should end in "DeathTest"
{
    KSParticle tInitialParticle;
    bool tResult = true;
    auto* tTerminator = new KSTermMaxLength();
    tTerminator->SetLength(10.);
    tTerminator->SetLength(-1.);
    ASSERT_ANY_THROW(tTerminator->CalculateTermination(tInitialParticle, tResult));
}

TEST(KassiopeiaTerminatorDeathTest, KSTermMaxLongEnergy)  // test case name should end in "DeathTest"
{
    KSParticle tInitialParticle;
    bool tResult = true;
    auto* tTerminator = new KSTermMaxLongEnergy();
    tTerminator->SetMaxLongEnergy(10.);
    tTerminator->SetMaxLongEnergy(-1.);
    ASSERT_ANY_THROW(tTerminator->CalculateTermination(tInitialParticle, tResult));
}

TEST(KassiopeiaTerminatorDeathTest, KSTermMaxR)  // test case name should end in "DeathTest"
{
    KSParticle tInitialParticle;
    bool tResult = true;
    auto* tTerminator = new KSTermMaxR();
    tTerminator->SetMaxR(10.);
    tTerminator->SetMaxR(-1.);
    ASSERT_ANY_THROW(tTerminator->CalculateTermination(tInitialParticle, tResult));
}

/* TODO there is no death test for KSTermMaxSteps */

TEST(KassiopeiaTerminatorDeathTest, KSTermMaxTime)  // test case name should end in "DeathTest"
{
    KSParticle tInitialParticle;
    bool tResult = true;
    auto* tTerminator = new KSTermMaxTime();
    tTerminator->SetTime(10.);
    tTerminator->SetTime(-1.);
    ASSERT_ANY_THROW(tTerminator->CalculateTermination(tInitialParticle, tResult));
}

/* TODO there is no death test for KSTermMaxZ */

TEST(KassiopeiaTerminatorDeathTest, KSTermMinEnergy)  // test case name should end in "DeathTest"
{
    KSParticle tInitialParticle;
    bool tResult = true;
    auto* tTerminator = new KSTermMinEnergy();
    tTerminator->SetMinEnergy(10.);
    tTerminator->SetMinEnergy(-1.);
    ASSERT_ANY_THROW(tTerminator->CalculateTermination(tInitialParticle, tResult));
}

TEST(KassiopeiaTerminatorDeathTest, KSTermMinLongEnergy)  // test case name should end in "DeathTest"
{
    KSParticle tInitialParticle;
    bool tResult = true;
    auto* tTerminator = new KSTermMinLongEnergy();
    tTerminator->SetMinLongEnergy(10.);
    tTerminator->SetMinLongEnergy(-1.);
    ASSERT_ANY_THROW(tTerminator->CalculateTermination(tInitialParticle, tResult));
}

TEST(KassiopeiaTerminatorDeathTest, KSTermMinR)  // test case name should end in "DeathTest"
{
    KSParticle tInitialParticle;
    bool tResult = true;
    auto* tTerminator = new KSTermMinR();
    tTerminator->SetMinR(10.);
    tTerminator->SetMinR(-1.);
    ASSERT_ANY_THROW(tTerminator->CalculateTermination(tInitialParticle, tResult));
}

/* TODO there is no death test for KSTermMinZ */

TEST(KassiopeiaTerminatorDeathTest, KSTermTrapped)  // test case name should end in "DeathTest"
{
    KSParticle tInitialParticle;
    bool tResult = true;
    auto* tTerminator = new KSTermTrapped();
    tTerminator->SetMaxTurns(10);
    tTerminator->SetMaxTurns(-1);
    ASSERT_ANY_THROW(tTerminator->CalculateTermination(tInitialParticle, tResult));
}
