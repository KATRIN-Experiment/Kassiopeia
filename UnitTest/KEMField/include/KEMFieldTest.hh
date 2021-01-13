#ifndef UNIT_TEST_KEMFIELDTEST_HH
#define UNIT_TEST_KEMFIELDTEST_HH

#include "UnitTest.h"

#include "gtest/gtest.h"

#ifdef KEMFIELD_USE_MPI
#include "KMPIInterface.hh"
#define MPI_SINGLE_PROCESS if (KEMField::KMPIInterface::GetInstance()->GetProcess() == 0)
#else
#define MPI_SINGLE_PROCESS
#endif

#ifdef KEMFIELD_USE_PETSC
#include "KPETScInterface.hh"
#endif

/* Use fixture to set up tracks used in several test cases */
class KEMFieldTest : public TimeoutTest
{
  protected:
    void SetUp() override
    {
        TimeoutTest::SetUp();

#ifdef KEMFIELD_USE_PETSC
        KEMField::KPETScInterface::GetInstance()->Initialize(0, nullptr);
#elif KEMFIELD_USE_MPI
        KEMField::KMPIInterface::GetInstance()->Initialize(0, nullptr);
#endif
    }

    void TearDown() override
    {
        // we cannot call Finalize() here because it will mess up subsequent tests
#ifdef KEMFIELD_USE_PETSC
        //KEMField::KPETScInterface::GetInstance()->Finalize();
#elif KEMFIELD_USE_MPI
        //KEMField::KMPIInterface::GetInstance()->Finalize();
#endif

        TimeoutTest::TearDown();
    }
};

#endif /* UNIT_TEST_KEMFIELDTEST_HH */
