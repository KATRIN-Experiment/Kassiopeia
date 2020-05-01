/*
 * KKrylovFactoryTest.cc
 *
 *  Created on: 13 Aug 2015
 *      Author: wolfgang
 */

#include "KEMSimpleException.hh"
#include "KKrylovSolverFactory.hh"
#include "KrylovFactoryFixture.hh"

namespace KEMField
{

TEST_F(KrylovFactoryFixture, DefaultToGMRES)
{
    ElectricSolverPtr solver = KBuildKrylovSolver<ElectricType>(fConfig, fA);
    ASSERT_TRUE(dynamic_cast<ElectricGMRES*>(&(*solver)));
    ASSERT_FALSE(dynamic_cast<ElectricBiCGSTAB*>(&(*solver)));
}

TEST_F(KrylovFactoryFixture, GiveBiCGSTAB)
{
    fConfig.SetSolverName("bicgstab");
    ElectricSolverPtr solver = KBuildKrylovSolver<ElectricType>(fConfig, fA);
    ASSERT_TRUE(dynamic_cast<ElectricBiCGSTAB*>(&(*solver)));
}

TEST_F(KrylovFactoryFixture, FailOnUnknownSolver)
{
    fConfig.SetSolverName("unknown");
    try {
        KBuildKrylovSolver<ElectricType>(fConfig, fA);
        ASSERT_FALSE("Should have thrown exception.");
    }
    catch (KEMSimpleException& e) {
        return;
    }
    ASSERT_FALSE("Should have caught exception.");
}

TEST_F(KrylovFactoryFixture, WithPreconDefaultToPGMRES)
{
    ElectricSolverPtr solver = KBuildKrylovSolver<ElectricType>(fConfig, fA, fP);
    ASSERT_TRUE(dynamic_cast<ElectricPGMRES*>(&(*solver)));
}

TEST_F(KrylovFactoryFixture, GivePBiCGSTAB)
{
    fConfig.SetSolverName("bicgstab");
    ElectricSolverPtr solver = KBuildKrylovSolver<ElectricType>(fConfig, fA, fP);
    ASSERT_TRUE(dynamic_cast<ElectricPBiCGSTAB*>(&(*solver)));
}


} /* namespace KEMField */
