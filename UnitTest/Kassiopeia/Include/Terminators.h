#ifndef UNIT_TEST_TERMINATORS_H
#define UNIT_TEST_TERMINATORS_H

#include "KSExampleFields.h"
#include "KSParticle.h"
#include "UnitTest.h"

#include "gtest/gtest.h"

using namespace Kassiopeia;

class KassiopeiaTerminatorTest : public TimeoutTest
{
  protected:
    void SetUp() override
    {
        TimeoutTest::SetUp();

        fInitialParticle = new KSParticle();
        fFinalParticle = new KSParticle();
        fParticles = new KSParticleQueue();
        fElectricField = MakeConstantElectricField(KGeoBag::KThreeVector(0., 0., 0.));
        fMagneticField = MakeConstantMagneticField(KGeoBag::KThreeVector(0., 0., 1.));
        ASSERT_PTR(fInitialParticle);
        ASSERT_PTR(fFinalParticle);
        ASSERT_PTR(fParticles);
        ASSERT_PTR(fElectricField);
        ASSERT_PTR(fMagneticField);

        fInitialParticle->SetElectricFieldCalculator(fElectricField);
        fInitialParticle->SetMagneticFieldCalculator(fMagneticField);
    }

    void TearDown() override
    {
        delete fInitialParticle;
        delete fFinalParticle;
        delete fParticles;
        delete fElectricField;
        delete fMagneticField;

        TimeoutTest::TearDown();
    }

    KSParticle* fInitialParticle;
    KSParticle* fFinalParticle;
    KSParticleQueue* fParticles;
    KSElectricField* fElectricField;
    KSMagneticField* fMagneticField;
};


#endif /* UNIT_TEST_TERMINATORS_H */
