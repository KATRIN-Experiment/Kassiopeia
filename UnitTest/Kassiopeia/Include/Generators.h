#ifndef UNIT_TEST_GENERATORS_H
#define UNIT_TEST_GENERATORS_H

#include "KSParticleFactory.h"
#include "UnitTest.h"

#include "gtest/gtest.h"
#include <vector>

using namespace Kassiopeia;

/* Use fixture to set up tracks used in several test cases */
class KassiopeiaGeneratorTest : public TimeoutTest
{
  protected:
    virtual void SetUp()
    {
        TimeoutTest::SetUp();

        fNTests = 1000;  // generate 1000 values in each test
    }

    virtual void TearDown()
    {
        fValues.clear();

        TimeoutTest::TearDown();
    }

    std::vector<double> fValues;
    unsigned int fNTests;
};

/* Use fixture to set up tracks used in several test cases */
class KassiopeiaCompositeGeneratorTest : public TimeoutTest
{
  protected:
    virtual void SetUp()
    {
        TimeoutTest::SetUp();

        fNTests = 100;  // generate 100 particles in each test

        // generate list of particles to be diced
        fParticles = new KSParticleQueue();
        ASSERT_PTR(fParticles);
        for (unsigned int i = 0; i < fNTests; i++)
            fParticles->push_back(new KSParticle());
    }

    virtual void TearDown()
    {
        ASSERT_PTR(fParticles);
        for (KSParticleIt tParticleIt = fParticles->begin(); tParticleIt != fParticles->end(); tParticleIt++)
            delete (*tParticleIt);
        delete fParticles;

        TimeoutTest::TearDown();
    }

    KSParticleQueue* fParticles;
    unsigned int fNTests;
};

#endif /* UNIT_TEST_GENERATORS_H */
