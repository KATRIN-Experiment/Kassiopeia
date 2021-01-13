/**
 * Unit testing for Kassiopeia generator classes
 * @author J. Behrens
 *
 * This file contains a unit tests for most of Kassiopeia's generator classes.
 * All tests should be grouped together in a meaningful way, and should use
 * a fixture class which is derived from TimeoutTest. Please keep unit tests
 * as simple as possible (think of a minimal example of how to use a certain
 * class). It is a good idea to also include a death-test for each class.
 *
 * See the official GoogleTest pages for more info:
 *   https://code.google.com/p/googletest/wiki/Primer
 *   https://code.google.com/p/googletest/wiki/AdvancedGuide
 */

#include "Generators.h"

#include "KSParticleFactory.h"

// Generators
#include "KSGenConversion.h"
#include "KSGenValueAngleSpherical.h"
#include "KSGenValueFix.h"
#include "KSGenValueFormula.h"
#include "KSGenValueGauss.h"
#include "KSGenValueSet.h"
#include "KSGenValueUniform.h"
//#include "KSGenRelaxation.h"  /* TODO not tested yet */
#include "KSGenShakeOff.h"
//#include "KSGeneratorTimePix.h"  /* TODO not tested yet */

#include "KSGenPositionSpaceRandom.h"
#include "KSGenPositionSurfaceRandom.h"

// Composite Generators
#include "KGCylinderSpace.hh"
#include "KGExtrudedCircleSpace.hh"
#include "KGExtrudedCircleSurface.hh"
#include "KSGenDirectionSphericalComposite.h"
#include "KSGenDirectionSurfaceComposite.h"
#include "KSGenEnergyComposite.h"
#include "KSGenEnergyKryptonEvent.h"
#include "KSGenEnergyRadonEvent.h"
#include "KSGenGeneratorComposite.h"
#include "KSGenPositionCylindricalComposite.h"
#include "KSGenPositionRectangularComposite.h"
#include "KSGenTimeComposite.h"

using namespace KGeoBag;
using namespace Kassiopeia;


//////////////////////////////////////////////////////////////////////////////
// Generators Unit Testing
//////////////////////////////////////////////////////////////////////////////

TEST_F(KassiopeiaGeneratorTest, KSGenValueFix)
{
    ASSERT_EQ(fValues.size(), 0UL);

    auto* tGenerator = new KSGenValueFix();
    ASSERT_PTR(tGenerator);
    tGenerator->SetValue(1);

    for (unsigned int i = 0; i < fNTests; i++) {
        tGenerator->DiceValue(fValues);
        ASSERT_EQ(fValues.size(), i + 1);
        EXPECT_EQ(fValues[i], 1.);
    }

    delete tGenerator;
}

//////////////////////////////////////////////////////////////////////////////

TEST_F(KassiopeiaGeneratorTest, KSGenValueFormula)
{
    ASSERT_EQ(fValues.size(), 0UL);

    auto* tGenerator = new KSGenValueFormula();
    ASSERT_PTR(tGenerator);
    tGenerator->SetValueMin(-1);
    tGenerator->SetValueMax(1);
    tGenerator->SetValueFormula("1-x^2");  // inverted parabola
    tGenerator->Initialize();

    for (unsigned int i = 0; i < fNTests; i++) {
        tGenerator->DiceValue(fValues);
        ASSERT_EQ(fValues.size(), i + 1);
        EXPECT_GE(fValues[i], -1.);
        EXPECT_LE(fValues[i], 1.);
    }

    tGenerator->Deinitialize();
    delete tGenerator;
}

//////////////////////////////////////////////////////////////////////////////

TEST_F(KassiopeiaGeneratorTest, KSGenValueGauss)
{
    ASSERT_EQ(fValues.size(), 0UL);

    auto* tGenerator = new KSGenValueGauss();
    ASSERT_PTR(tGenerator);
    tGenerator->SetValueMin(-1);
    tGenerator->SetValueMax(1);
    tGenerator->SetValueMean(0);
    tGenerator->SetValueSigma(0.2);

    for (unsigned int i = 0; i < fNTests; i++) {
        tGenerator->DiceValue(fValues);
        ASSERT_EQ(fValues.size(), i + 1);
        EXPECT_GE(fValues[i], -1.);
        EXPECT_LE(fValues[i], 1.);
    }

    delete tGenerator;
}

//////////////////////////////////////////////////////////////////////////////

TEST_F(KassiopeiaGeneratorTest, KSGenValueAngleSpherical)
{
    ASSERT_EQ(fValues.size(), 0UL);

    auto* tGenerator = new KSGenValueAngleSpherical();
    ASSERT_PTR(tGenerator);
    tGenerator->SetAngleMin(0);
    tGenerator->SetAngleMax(180);

    for (unsigned int i = 0; i < fNTests; i++) {
        tGenerator->DiceValue(fValues);
        ASSERT_EQ(fValues.size(), i + 1);
        EXPECT_GE(fValues[i], 0.);
        EXPECT_LE(fValues[i], 180.);
    }

    delete tGenerator;
}

//////////////////////////////////////////////////////////////////////////////

TEST_F(KassiopeiaGeneratorTest, KSGenValueSet)
{
    ASSERT_EQ(fValues.size(), 0UL);

    auto* tGenerator = new KSGenValueSet();
    ASSERT_PTR(tGenerator);
    tGenerator->SetValueStart(-1);
    tGenerator->SetValueStop(1);
    tGenerator->SetValueCount(2);

    for (unsigned int i = 0; i < fNTests; i++) {
        tGenerator->DiceValue(fValues);
        ASSERT_EQ(fValues.size(), 2 * i + 2);
        EXPECT_EQ(fValues[2 * i], -1.);
        EXPECT_EQ(fValues[2 * i + 1], 1.);
    }

    delete tGenerator;
}

//////////////////////////////////////////////////////////////////////////////

TEST_F(KassiopeiaGeneratorTest, KSGenValueUniform)
{
    ASSERT_EQ(fValues.size(), 0UL);

    auto* tGenerator = new KSGenValueUniform();
    ASSERT_PTR(tGenerator);
    tGenerator->SetValueMin(-1);
    tGenerator->SetValueMax(1);

    for (unsigned int i = 0; i < fNTests; i++) {
        tGenerator->DiceValue(fValues);
        ASSERT_EQ(fValues.size(), i + 1);
        EXPECT_GE(fValues[i], -1.);
        EXPECT_LE(fValues[i], 1.);
    }

    delete tGenerator;
}

//////////////////////////////////////////////////////////////////////////////

TEST_F(KassiopeiaGeneratorTest, KSGenConversion_Kr83)
{
    ASSERT_EQ(fValues.size(), 0UL);
    vector<int> tVacancies;

    auto* tGenerator = new KSGenConversion();
    ASSERT_PTR(tGenerator);
    tGenerator->SetForceCreation(true);
    tGenerator->Initialize(83);

    for (unsigned int i = 0; i < fNTests; i++) {
        tGenerator->CreateCE(tVacancies, fValues);
        ASSERT_EQ(fValues.size(), tVacancies.size());
    }
    ASSERT_EQ(fValues.size(), fNTests);

    for (unsigned int i = 0; i < fValues.size(); i++) {
        EXPECT_GT(fValues[i], 0.);
        EXPECT_LE(fValues[i], 32137.5);
        EXPECT_GE(tVacancies[i], 1);
        EXPECT_LE(tVacancies[i], 12);
    }

    delete tGenerator;
}

//////////////////////////////////////////////////////////////////////////////

TEST_F(KassiopeiaGeneratorTest, KSGenConversion_Rn219)
{
    ASSERT_EQ(fValues.size(), 0UL);
    vector<int> tVacancies;

    auto* tGenerator = new KSGenConversion();
    ASSERT_PTR(tGenerator);
    tGenerator->SetForceCreation(true);
    tGenerator->Initialize(219);

    for (unsigned int i = 0; i < fNTests; i++) {
        tGenerator->CreateCE(tVacancies, fValues);
        ASSERT_EQ(fValues.size(), tVacancies.size());
    }
    ASSERT_EQ(fValues.size(), fNTests);

    for (unsigned int i = 0; i < fValues.size(); i++) {
        EXPECT_GT(fValues[i], 0.);
        EXPECT_LE(fValues[i], 500660.);
        EXPECT_GE(tVacancies[i], 1);
        EXPECT_LE(tVacancies[i], 10);
    }

    delete tGenerator;
}

//////////////////////////////////////////////////////////////////////////////

TEST_F(KassiopeiaGeneratorTest, KSGenConversion_Rn220)
{
    ASSERT_EQ(fValues.size(), 0UL);
    vector<int> tVacancies;

    auto* tGenerator = new KSGenConversion();
    ASSERT_PTR(tGenerator);
    tGenerator->SetForceCreation(true);
    tGenerator->Initialize(220);

    for (unsigned int i = 0; i < fNTests; i++) {
        tGenerator->CreateCE(tVacancies, fValues);
        ASSERT_EQ(fValues.size(), tVacancies.size());
    }
    ASSERT_EQ(fValues.size(), fNTests);

    for (unsigned int i = 0; i < fValues.size(); i++) {
        EXPECT_GT(fValues[i], 0.);
        EXPECT_LE(fValues[i], 534000.);
        EXPECT_GE(tVacancies[i], 1);
        EXPECT_LE(tVacancies[i], 2);
    }

    delete tGenerator;
}

//////////////////////////////////////////////////////////////////////////////

/* TODO KSGenRelaxation not tested because I don't know how
 * ... but we still test the full Kr/Rn energy creators! */

//////////////////////////////////////////////////////////////////////////////

TEST_F(KassiopeiaGeneratorTest, KSGenShakeOff_Rn)
{
    ASSERT_EQ(fValues.size(), 0UL);
    vector<int> tVacancies;

    auto* tGenerator = new KSGenShakeOff();
    ASSERT_PTR(tGenerator);
    tGenerator->SetForceCreation(false);

    for (unsigned int i = 0; i < fNTests; i++) {
        tGenerator->CreateSO(tVacancies, fValues);
        ASSERT_EQ(fValues.size(), tVacancies.size());
    }
    ASSERT_GT(fValues.size(), 0UL);

    for (unsigned long i = 0; i < fValues.size(); i++) {
        EXPECT_GT(fValues[i], 0.);
        EXPECT_LE(fValues[i], 93106.);
        EXPECT_GE(tVacancies[i], 1);
        EXPECT_LE(tVacancies[i], 9);
    }

    delete tGenerator;
}

//////////////////////////////////////////////////////////////////////////////
// DEATH TESTS ///////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

TEST(KassiopeiaGeneratorDeathTest, KSGenConversion)  // test case name should end in "DeathTest"
{
    // should fail because isotope 0 is not defined
    auto* tGenerator = new KSGenConversion();
    ASSERT_PTR(tGenerator);
    ASSERT_ANY_THROW(tGenerator->Initialize(0));
}


//////////////////////////////////////////////////////////////////////////////
// Composite Generators Unit Testing
//////////////////////////////////////////////////////////////////////////////

TEST_F(KassiopeiaCompositeGeneratorTest, KSGenDirectionSphericalComposite)
{
    ASSERT_EQ(fParticles->size(), fNTests);

    auto* tPhiGenerator = new KSGenValueAngleSpherical();
    ASSERT_PTR(tPhiGenerator);
    tPhiGenerator->SetAngleMin(0);
    tPhiGenerator->SetAngleMax(180);

    auto* tThetaGenerator = new KSGenValueAngleSpherical();
    ASSERT_PTR(tThetaGenerator);
    tThetaGenerator->SetAngleMin(0);
    tThetaGenerator->SetAngleMax(90);

    auto* tCompositeGenerator = new KSGenDirectionSphericalComposite();
    ASSERT_PTR(tCompositeGenerator);
    tCompositeGenerator->SetPhiValue(tPhiGenerator);
    tCompositeGenerator->SetThetaValue(tThetaGenerator);

    tCompositeGenerator->SetName("test");
    EXPECT_STRING_EQ(tCompositeGenerator->GetName(), "test");

    //    tCompositeGenerator->Initialize();
    tCompositeGenerator->Dice(fParticles);
    EXPECT_EQ(fParticles->size(), fNTests);

    for (unsigned int i = 0; i < fNTests; i++) {
        KThreeVector tMomentum = fParticles->at(i)->GetMomentum();
        EXPECT_GE(tMomentum.AzimuthalAngle(), 0.);
        EXPECT_LE(tMomentum.AzimuthalAngle(), 180.);
        EXPECT_GE(tMomentum.PolarAngle(), 0.);
        EXPECT_LE(tMomentum.PolarAngle(), 90.);
    }

    //    tCompositeGenerator->Deinitialize();

    delete tCompositeGenerator;
    delete tThetaGenerator;
    delete tPhiGenerator;
}

//////////////////////////////////////////////////////////////////////////////

TEST_F(KassiopeiaCompositeGeneratorTest, KSGenEnergyComposite)
{
    ASSERT_EQ(fParticles->size(), fNTests);

    auto* tEnergyGenerator = new KSGenValueUniform();
    ASSERT_PTR(tEnergyGenerator);
    tEnergyGenerator->SetValueMin(0);
    tEnergyGenerator->SetValueMax(1000);

    auto* tCompositeGenerator = new KSGenEnergyComposite();
    ASSERT_PTR(tCompositeGenerator);
    tCompositeGenerator->SetEnergyValue(tEnergyGenerator);

    tCompositeGenerator->SetName("test");
    EXPECT_STRING_EQ(tCompositeGenerator->GetName(), "test");

    //    tCompositeGenerator->Initialize();
    tCompositeGenerator->Dice(fParticles);
    EXPECT_EQ(fParticles->size(), fNTests);

    for (unsigned int i = 0; i < fNTests; i++) {
        double tKineticEnergy = fParticles->at(i)->GetKineticEnergy_eV();
        EXPECT_GE(tKineticEnergy, 0.);
        EXPECT_LE(tKineticEnergy, 1000.);
    }

    //    tCompositeGenerator->Deinitialize();

    delete tCompositeGenerator;
    delete tEnergyGenerator;
}

//////////////////////////////////////////////////////////////////////////////

TEST_F(KassiopeiaCompositeGeneratorTest, KSGenCompositePositionCylindrical)
{
    ASSERT_EQ(fParticles->size(), fNTests);

    auto* tRGenerator = new KSGenValueUniform();
    ASSERT_PTR(tRGenerator);
    tRGenerator->SetValueMin(0);
    tRGenerator->SetValueMax(0.5);

    auto* tPhiGenerator = new KSGenValueAngleSpherical();
    ASSERT_PTR(tPhiGenerator);
    tPhiGenerator->SetAngleMin(0);
    tPhiGenerator->SetAngleMax(180);

    auto* tZGenerator = new KSGenValueUniform();
    ASSERT_PTR(tZGenerator);
    tZGenerator->SetValueMin(0);
    tZGenerator->SetValueMax(1);

    auto* tCompositeGenerator = new KSGenPositionCylindricalComposite();
    ASSERT_PTR(tCompositeGenerator);
    tCompositeGenerator->SetRValue(tRGenerator);
    tCompositeGenerator->SetPhiValue(tPhiGenerator);
    tCompositeGenerator->SetZValue(tZGenerator);
    tCompositeGenerator->SetOrigin(KThreeVector(0, 0, -0.5));

    tCompositeGenerator->SetName("test");
    EXPECT_STRING_EQ(tCompositeGenerator->GetName(), "test");

    //    tCompositeGenerator->Initialize();
    tCompositeGenerator->Dice(fParticles);
    EXPECT_EQ(fParticles->size(), fNTests);

    for (unsigned int i = 0; i < fNTests; i++) {
        KThreeVector tPosition = fParticles->at(i)->GetPosition();
        EXPECT_GE(tPosition.Z(), -0.5);
        EXPECT_LE(tPosition.Z(), 0.5);
        EXPECT_GE(tPosition.Perp(), 0);
        EXPECT_LE(tPosition.Perp(), 0.5);
        EXPECT_GE(tPosition.AzimuthalAngle(), 0.);
        EXPECT_LE(tPosition.AzimuthalAngle(), 180.);
    }

    //    tCompositeGenerator->Deinitialize();

    delete tCompositeGenerator;
    delete tRGenerator;
    delete tPhiGenerator;
    delete tZGenerator;
}

//////////////////////////////////////////////////////////////////////////////

TEST_F(KassiopeiaCompositeGeneratorTest, KSGenCompositePositionRectangular)
{
    ASSERT_EQ(fParticles->size(), fNTests);

    auto* tXGenerator = new KSGenValueUniform();
    ASSERT_PTR(tXGenerator);
    tXGenerator->SetValueMin(0);
    tXGenerator->SetValueMax(1);

    auto* tYGenerator = new KSGenValueAngleSpherical();
    ASSERT_PTR(tYGenerator);
    tYGenerator->SetAngleMin(0);
    tYGenerator->SetAngleMax(1);

    auto* tZGenerator = new KSGenValueUniform();
    ASSERT_PTR(tZGenerator);
    tZGenerator->SetValueMin(0);
    tZGenerator->SetValueMax(1);

    auto* tCompositeGenerator = new KSGenPositionRectangularComposite();
    ASSERT_PTR(tCompositeGenerator);
    tCompositeGenerator->SetXValue(tXGenerator);
    tCompositeGenerator->SetYValue(tYGenerator);
    tCompositeGenerator->SetZValue(tZGenerator);
    tCompositeGenerator->SetOrigin(KThreeVector(0, 0, -0.5));

    tCompositeGenerator->SetName("test");
    EXPECT_STRING_EQ(tCompositeGenerator->GetName(), "test");

    //    tCompositeGenerator->Initialize();
    tCompositeGenerator->Dice(fParticles);
    EXPECT_EQ(fParticles->size(), fNTests);

    for (unsigned int i = 0; i < fNTests; i++) {
        KThreeVector tPosition = fParticles->at(i)->GetPosition();
        EXPECT_GE(tPosition.X(), 0);
        EXPECT_LE(tPosition.X(), 1);
        EXPECT_GE(tPosition.Y(), 0);
        EXPECT_LE(tPosition.Y(), 1);
        EXPECT_GE(tPosition.Z(), -0.5);
        EXPECT_LE(tPosition.Z(), 0.5);
    }

    //    tCompositeGenerator->Deinitialize();

    delete tCompositeGenerator;
    delete tXGenerator;
    delete tYGenerator;
    delete tZGenerator;
}

//////////////////////////////////////////////////////////////////////////////

TEST_F(KassiopeiaCompositeGeneratorTest, KSGenPositionSpaceRandom)
{
    ASSERT_EQ(fParticles->size(), fNTests);

    auto* tCylinder = new KGeoBag::KGCylinderSpace();
    ASSERT_PTR(tCylinder);
    tCylinder->Z1(0.);
    tCylinder->Z2(1.);
    tCylinder->R(0.5);

    auto* tSpace = new KGeoBag::KGSpace();
    tSpace->Volume(std::shared_ptr<KGeoBag::KGVolume>(tCylinder));

    auto* tPositionGenerator = new KSGenPositionSpaceRandom();
    ASSERT_PTR(tPositionGenerator);
    tPositionGenerator->AddSpace(tSpace);

    tPositionGenerator->SetName("test");
    EXPECT_STRING_EQ(tPositionGenerator->GetName(), "test");

    tPositionGenerator->Initialize();
    tPositionGenerator->Dice(fParticles);
    EXPECT_EQ(fParticles->size(), fNTests);

    for (unsigned int i = 0; i < fNTests; i++) {
        KThreeVector tPosition = fParticles->at(i)->GetPosition();
        EXPECT_GE(tPosition.Perp(), 0.);
        EXPECT_LE(tPosition.Perp(), 0.5);
        EXPECT_GE(tPosition.Z(), 0.);
        EXPECT_LE(tPosition.Z(), 1.);
    }

    tPositionGenerator->Deinitialize();

    delete tPositionGenerator;
    delete tCylinder;
}

//////////////////////////////////////////////////////////////////////////////

TEST_F(KassiopeiaCompositeGeneratorTest, KSGenPositionSurfaceRandom)
{
    ASSERT_EQ(fParticles->size(), fNTests);

    auto* tCylinder = new KGeoBag::KGCylinderSurface();
    ASSERT_PTR(tCylinder);
    tCylinder->Z1(0.);
    tCylinder->Z2(1.);
    tCylinder->R(0.5);

    auto* tSurface = new KGeoBag::KGSurface();
    tSurface->Area(std::shared_ptr<KGeoBag::KGArea>(tCylinder));

    auto* tPositionGenerator = new KSGenPositionSurfaceRandom();
    ASSERT_PTR(tPositionGenerator);
    tPositionGenerator->AddSurface(tSurface);

    tPositionGenerator->SetName("test");
    EXPECT_STRING_EQ(tPositionGenerator->GetName(), "test");

    tPositionGenerator->Initialize();
    tPositionGenerator->Dice(fParticles);
    EXPECT_EQ(fParticles->size(), fNTests);

    for (unsigned int i = 0; i < fNTests; i++) {
        KThreeVector tPosition = fParticles->at(i)->GetPosition();
        EXPECT_NEAR(tPosition.Perp(), 0.5, 1e-6);
        EXPECT_GE(tPosition.Z(), 0.);
        EXPECT_LE(tPosition.Z(), 1.);
    }

    tPositionGenerator->Deinitialize();

    delete tPositionGenerator;
    delete tCylinder;
}
//////////////////////////////////////////////////////////////////////////////

TEST_F(KassiopeiaCompositeGeneratorTest, KSGenTimeComposite)
{
    ASSERT_EQ(fParticles->size(), fNTests);

    auto* tTimeGenerator = new KSGenValueUniform();
    ASSERT_PTR(tTimeGenerator);
    tTimeGenerator->SetValueMin(0);
    tTimeGenerator->SetValueMax(1e-3);

    auto* tCompositeGenerator = new KSGenTimeComposite();
    ASSERT_PTR(tCompositeGenerator);
    tCompositeGenerator->SetTimeValue(tTimeGenerator);

    tCompositeGenerator->SetName("test");
    EXPECT_STRING_EQ(tCompositeGenerator->GetName(), "test");

    tCompositeGenerator->Initialize();
    tCompositeGenerator->Dice(fParticles);
    EXPECT_EQ(fParticles->size(), fNTests);

    for (unsigned int i = 0; i < fNTests; i++) {
        double tTime = fParticles->at(i)->GetTime();
        EXPECT_GE(tTime, 0.);
        EXPECT_LE(tTime, 1e-3);
    }

    tCompositeGenerator->Deinitialize();

    delete tCompositeGenerator;
    delete tTimeGenerator;
}

//////////////////////////////////////////////////////////////////////////////

TEST_F(KassiopeiaCompositeGeneratorTest, KSGenGeneratorComposite)
{
    ASSERT_EQ(fParticles->size(), fNTests);

    auto* tEnergyGenerator = new KSGenValueUniform();
    ASSERT_PTR(tEnergyGenerator);
    tEnergyGenerator->SetValueMin(0);
    tEnergyGenerator->SetValueMax(1000);

    auto* tPhiGenerator = new KSGenValueAngleSpherical();
    ASSERT_PTR(tPhiGenerator);
    tPhiGenerator->SetAngleMin(0);
    tPhiGenerator->SetAngleMax(180);

    auto* tThetaGenerator = new KSGenValueAngleSpherical();
    ASSERT_PTR(tThetaGenerator);
    tThetaGenerator->SetAngleMin(0);
    tThetaGenerator->SetAngleMax(90);

    auto* tRGenerator = new KSGenValueUniform();
    ASSERT_PTR(tRGenerator);
    tRGenerator->SetValueMin(0);
    tRGenerator->SetValueMax(0.5);

    auto* tZGenerator = new KSGenValueUniform();
    ASSERT_PTR(tZGenerator);
    tZGenerator->SetValueMin(0);
    tZGenerator->SetValueMax(1);

    auto* tTimeGenerator = new KSGenValueUniform();
    ASSERT_PTR(tTimeGenerator);
    tTimeGenerator->SetValueMin(0);
    tTimeGenerator->SetValueMax(1e-3);

    auto* tCompositeEnergyGenerator = new KSGenEnergyComposite();
    ASSERT_PTR(tCompositeEnergyGenerator);
    tCompositeEnergyGenerator->SetEnergyValue(tEnergyGenerator);

    auto* tCompositeDirectionGenerator = new KSGenDirectionSphericalComposite();
    ASSERT_PTR(tCompositeDirectionGenerator);
    tCompositeDirectionGenerator->SetPhiValue(tPhiGenerator);
    tCompositeDirectionGenerator->SetThetaValue(tThetaGenerator);

    auto* tCompositePositionGenerator = new KSGenPositionCylindricalComposite();
    tCompositePositionGenerator->SetRValue(tRGenerator);
    ASSERT_PTR(tCompositePositionGenerator);
    tCompositePositionGenerator->SetPhiValue(tPhiGenerator);
    tCompositePositionGenerator->SetZValue(tZGenerator);
    tCompositePositionGenerator->SetOrigin(KThreeVector(0, 0, -0.5));

    auto* tCompositeTimeGenerator = new KSGenTimeComposite();
    ASSERT_PTR(tCompositeTimeGenerator);
    tCompositeTimeGenerator->SetTimeValue(tTimeGenerator);

    auto* tCompositePidGenerator = new KSGenValueFix();
    ASSERT_PTR(tCompositePidGenerator);
    tCompositePidGenerator->SetValue(11);

    auto* tCompositeGenerator = new KSGenGeneratorComposite();
    ASSERT_PTR(tCompositeGenerator);
    tCompositeGenerator->AddCreator(tCompositeEnergyGenerator);
    tCompositeGenerator->AddCreator(tCompositeDirectionGenerator);
    tCompositeGenerator->AddCreator(tCompositePositionGenerator);
    tCompositeGenerator->AddCreator(tCompositeTimeGenerator);
    tCompositeGenerator->SetPid(tCompositePidGenerator);

    tCompositeGenerator->SetName("test");
    EXPECT_STRING_EQ(tCompositeGenerator->GetName(), "test");

    tCompositeGenerator->Initialize();
    fParticles->clear();  // KSGenCompositeGenerator automatically fills in new particles
    for (unsigned int i = 0; i < fNTests; i++)
        tCompositeGenerator->ExecuteGeneration(*fParticles);  // why is this reference and not pointer?
    EXPECT_EQ(fParticles->size(), fNTests);

    for (unsigned int i = 0; i < fNTests; i++) {
        double tKineticEnergy = fParticles->at(i)->GetKineticEnergy_eV();
        KThreeVector tPosition = fParticles->at(i)->GetPosition();
        KThreeVector tMomentum = fParticles->at(i)->GetMomentum();
        double tTime = fParticles->at(i)->GetTime();
        EXPECT_GE(tKineticEnergy, 0.);
        EXPECT_LE(tKineticEnergy, 1000.);
        EXPECT_GE(tPosition.Z(), -0.5);
        EXPECT_LE(tPosition.Z(), 0.5);
        EXPECT_GE(tPosition.Perp(), 0.);
        EXPECT_LE(tPosition.Perp(), 0.5);
        EXPECT_GE(tPosition.AzimuthalAngle(), 0.);
        EXPECT_LE(tPosition.AzimuthalAngle(), 180.);
        EXPECT_GE(tMomentum.AzimuthalAngle(), 0.);
        EXPECT_LE(tMomentum.AzimuthalAngle(), 180.);
        EXPECT_GE(tMomentum.PolarAngle(), 0.);
        EXPECT_LE(tMomentum.PolarAngle(), 90.);
        EXPECT_GE(tTime, 0.);
        EXPECT_LE(tTime, 1e-3);
    }

    tCompositeGenerator->Deinitialize();

    delete tCompositeGenerator;
    delete tCompositeEnergyGenerator;
    delete tCompositeDirectionGenerator;
    delete tCompositePositionGenerator;
    delete tCompositeTimeGenerator;
    delete tCompositePidGenerator;
    delete tEnergyGenerator;
    delete tPhiGenerator;
    delete tThetaGenerator;
    delete tRGenerator;
    delete tZGenerator;
    delete tTimeGenerator;
}

//////////////////////////////////////////////////////////////////////////////

#if __TEST_IS_BROKEN

TEST_F(KassiopeiaCompositeGeneratorTest, KSGenEnergyKryptonEvent_Kr83)
{
    ASSERT_EQ(fParticles->size(), fNTests);

    KSGenEnergyKryptonEvent* tEnergyGenerator = new KSGenEnergyKryptonEvent();
    ASSERT_PTR(tEnergyGenerator);
    tEnergyGenerator->SetForceConversion(false);
    tEnergyGenerator->SetDoConversion(true);
    tEnergyGenerator->SetDoAuger(true);

    tEnergyGenerator->SetName("test");
    EXPECT_STRING_EQ(tEnergyGenerator->GetName(), "test");

    tEnergyGenerator->Initialize();

    unsigned int tCount = 0;
    for (unsigned int i = 0; i < fNTests; i++) {
        tEnergyGenerator->Dice(fParticles);  // KSGenEnergyRadonEvent clears the particle list
        ASSERT_GT(fParticles->size(), 0UL);
        for (KSParticleIt tIt = fParticles->begin(); tIt != fParticles->end(); tIt++) {
            double tKineticEnergy = (*tIt)->GetKineticEnergy_eV();
            EXPECT_GT(tKineticEnergy, 0.);
            EXPECT_LE(tKineticEnergy, 32137.5 + ROUND_ERROR_DOUBLE);
            tCount++;
        }
    }
    EXPECT_GE(tCount, fNTests);

    tEnergyGenerator->Deinitialize();

    delete tEnergyGenerator;
}

//////////////////////////////////////////////////////////////////////////////

TEST_F(KassiopeiaCompositeGeneratorTest, KSGenEnergyRadonEvent_Rn219)
{
    ASSERT_EQ(fParticles->size(), fNTests);

    KSGenEnergyRadonEvent* tEnergyGenerator = new KSGenEnergyRadonEvent();
    tEnergyGenerator->SetIsotope(219);
    tEnergyGenerator->SetForceConversion(false);
    tEnergyGenerator->SetForceShakeOff(false);
    tEnergyGenerator->SetDoConversion(true);
    tEnergyGenerator->SetDoShakeOff(true);
    tEnergyGenerator->SetDoAuger(true);

    tEnergyGenerator->SetName("test");
    EXPECT_STRING_EQ(tEnergyGenerator->GetName(), "test");

    tEnergyGenerator->Initialize();

    unsigned int tCount = 0;
    for (unsigned int i = 0; i < fNTests; i++) {
        tEnergyGenerator->Dice(fParticles);  // KSGenEnergyRadonEvent clears the particle list
        ASSERT_GT(fParticles->size(), 0UL);
        for (KSParticleIt tIt = fParticles->begin(); tIt != fParticles->end(); tIt++) {
            double tKineticEnergy = (*tIt)->GetKineticEnergy_eV();
            EXPECT_GT(tKineticEnergy, 0.);
            EXPECT_LE(tKineticEnergy, 500660. + ROUND_ERROR_DOUBLE);
            tCount++;
        }
    }
    EXPECT_GE(tCount, fNTests);

    tEnergyGenerator->Deinitialize();

    delete tEnergyGenerator;
}

//////////////////////////////////////////////////////////////////////////////

TEST_F(KassiopeiaCompositeGeneratorTest, KSGenEnergyRadonEvent_Rn220)
{
    ASSERT_EQ(fParticles->size(), fNTests);

    KSGenEnergyRadonEvent* tEnergyGenerator = new KSGenEnergyRadonEvent();
    tEnergyGenerator->SetIsotope(220);
    tEnergyGenerator->SetForceConversion(false);
    tEnergyGenerator->SetForceShakeOff(false);
    tEnergyGenerator->SetDoConversion(true);
    tEnergyGenerator->SetDoShakeOff(true);
    tEnergyGenerator->SetDoAuger(true);

    tEnergyGenerator->SetName("test");
    EXPECT_STRING_EQ(tEnergyGenerator->GetName(), "test");

    tEnergyGenerator->Initialize();

    unsigned int tCount = 0;
    for (unsigned int i = 0; i < fNTests; i++) {
        tEnergyGenerator->Dice(fParticles);  // KSGenEnergyRadonEvent clears the particle list
        ASSERT_GT(fParticles->size(), 0UL);
        for (KSParticleIt tIt = fParticles->begin(); tIt != fParticles->end(); tIt++) {
            double tKineticEnergy = (*tIt)->GetKineticEnergy_eV();
            EXPECT_GT(tKineticEnergy, 0.);
            EXPECT_LE(tKineticEnergy, 93106. + ROUND_ERROR_DOUBLE);
            tCount++;
        }
    }
    EXPECT_GE(tCount, fNTests);

    tEnergyGenerator->Deinitialize();

    delete tEnergyGenerator;
}

#endif  // __TEST_IS_BROKEN

//////////////////////////////////////////////////////////////////////////////
// DEATH TESTS ///////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

TEST(KassiopeiaCompositeGeneratorDeathTest,
     KSGenDirectionSphericalComposite)  // test case name should end in "DeathTest"
{
    KSParticleQueue tParticles;
    auto* tCompositeGenerator = new KSGenDirectionSphericalComposite();
    //    tCompositeGenerator->Initialize();
    ASSERT_ANY_THROW(tCompositeGenerator->Dice(&tParticles));
    //    tCompositeGenerator->Deinitialize();
}

TEST(KassiopeiaCompositeGeneratorDeathTest, KSGenEnergyComposite)  // test case name should end in "DeathTest"
{
    KSParticleQueue tParticles;
    auto* tCompositeGenerator = new KSGenEnergyComposite();
    //    tCompositeGenerator->Initialize();
    ASSERT_ANY_THROW(tCompositeGenerator->Dice(&tParticles));
    //    tCompositeGenerator->Deinitialize();
}

TEST(KassiopeiaCompositeGeneratorDeathTest,
     KSGenPositionCylindricalComposite)  // test case name should end in "DeathTest"
{
    KSParticleQueue tParticles;
    auto* tCompositeGenerator = new KSGenPositionCylindricalComposite();
    //    tCompositeGenerator->Initialize();
    ASSERT_ANY_THROW(tCompositeGenerator->Dice(&tParticles));
    //    tCompositeGenerator->Deinitialize();
}

TEST(KassiopeiaCompositeGeneratorDeathTest,
     KSGenPositionRectangularComposite)  // test case name should end in "DeathTest"
{
    KSParticleQueue tParticles;
    auto* tCompositeGenerator = new KSGenPositionRectangularComposite();
    //    tCompositeGenerator->Initialize();
    ASSERT_ANY_THROW(tCompositeGenerator->Dice(&tParticles));
    //    tCompositeGenerator->Deinitialize();
}

TEST(KassiopeiaCompositeGeneratorDeathTest, KSGenTimeComposite)  // test case name should end in "DeathTest"
{
    KSParticleQueue tParticles;
    auto* tCompositeGenerator = new KSGenTimeComposite();
    //    tCompositeGenerator->Initialize();
    ASSERT_ANY_THROW(tCompositeGenerator->Dice(&tParticles));
    //    tCompositeGenerator->Deinitialize();
}
