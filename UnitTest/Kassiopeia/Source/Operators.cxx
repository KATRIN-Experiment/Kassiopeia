/**
 * Unit testing for Kassiopeia's operator classes
 * @author J. Behrens
 *
 * This file contains a unit tests for most of Kassiopeia's operator classes.
 * All tests should be grouped together in a meaningful way, and should use
 * a fixture class which is derived from TimeoutTest. Please keep unit tests
 * as simple as possible (think of a minimal example of how to use a certain
 * class). It is a good idea to also include a death-test for each class.
 *
 * See the official GoogleTest pages for more info:
 *   https://code.google.com/p/googletest/wiki/Primer
 *   https://code.google.com/p/googletest/wiki/AdvancedGuide
 */

#include "Operators.h"

#include "KConst.h"

//#include "KSElectricField.h"
//#include "KSGenerator.h"
//#include "KSGeometry.h"
//#include "KSMagneticField.h"
#include "KSParticle.h"
#include "KSParticleFactory.h"
//#include "KSSpaceBuilder.h"
//#include "KSSpaceData.h"
//#include "KSSpaceInteraction.h"
//#include "KSSpaceNavigator.h"
//#include "KSSurfaceBuilder.h"
//#include "KSSurfaceData.h"
//#include "KSSurfaceInteraction.h"
//#include "KSSurfaceNavigator.h"
//#include "KSTerminator.h"
//#include "KSTrajectory.h"
//#include "KSWriter.h"

using namespace Kassiopeia;


/////////////////////////////////////////////////////////////////////////////
// Operators Unit Testing
/////////////////////////////////////////////////////////////////////////////

TEST_F(KassiopeiaOperatorsTest, KSParticle)
{
    auto* tOperator = new KSParticle();
    ASSERT_PTR(tOperator);

    std::string name = "test_label";

    tOperator->AddLabel(name);
    tOperator->ReleaseLabel(name);

    tOperator->SetParentRunId(1);
    EXPECT_EQ(tOperator->GetParentRunId(), 1);

    tOperator->SetParentEventId(1);
    EXPECT_EQ(tOperator->GetParentEventId(), 1);

    tOperator->SetParentTrackId(1);
    EXPECT_EQ(tOperator->GetParentTrackId(), 1);

    tOperator->SetParentStepId(1);
    EXPECT_EQ(tOperator->GetParentStepId(), 1);

    delete tOperator;
}

//////////////////////////////////////////////////////////////////////////////

TEST_F(KassiopeiaOperatorsTest, KSParticleFactory)
{
    KSParticleFactory* tFactory = &KSParticleFactory::GetInstance();
    ASSERT_PTR(tFactory);

    KSParticle* tGhost = tFactory->Create(0);
    ASSERT_PTR(tGhost);
    EXPECT_NEAR(tGhost->GetMass(), 0., ROUND_ERROR_DOUBLE);  // non-zero mass
    EXPECT_EQ(tGhost->GetCharge(), 0.);
    delete tGhost;

    KSParticle* tElectron = tFactory->Create(11);
    ASSERT_PTR(tElectron);
    EXPECT_EQ(tElectron->GetMass(), katrin::KConst::M_el_kg());
    EXPECT_EQ(tElectron->GetCharge(), -1. * katrin::KConst::Q());
    delete tElectron;

    KSParticle* tPositron = tFactory->Create(-11);
    ASSERT_PTR(tPositron);
    EXPECT_EQ(tPositron->GetMass(), katrin::KConst::M_el_kg());
    EXPECT_EQ(tPositron->GetCharge(), katrin::KConst::Q());
    delete tPositron;

    KSParticle* tMuMinus = tFactory->Create(13);
    ASSERT_PTR(tMuMinus);
    EXPECT_EQ(tMuMinus->GetMass(), katrin::KConst::M_mu_kg());
    EXPECT_EQ(tMuMinus->GetCharge(), -1. * katrin::KConst::Q());
    delete tMuMinus;

    KSParticle* tMuPlus = tFactory->Create(-13);
    ASSERT_PTR(tMuPlus);
    EXPECT_EQ(tMuPlus->GetMass(), katrin::KConst::M_mu_kg());
    EXPECT_EQ(tMuPlus->GetCharge(), katrin::KConst::Q());
    delete tMuPlus;

    KSParticle* tProton = tFactory->Create(2212);
    ASSERT_PTR(tProton);
    EXPECT_EQ(tProton->GetMass(), katrin::KConst::M_prot_kg());
    EXPECT_EQ(tProton->GetCharge(), katrin::KConst::Q());
    delete tProton;

    KSParticle* tAntiProton = tFactory->Create(-2212);
    ASSERT_PTR(tAntiProton);
    EXPECT_EQ(tAntiProton->GetMass(), katrin::KConst::M_prot_kg());
    EXPECT_EQ(tAntiProton->GetCharge(), -1. * katrin::KConst::Q());
    delete tAntiProton;

    KSParticle* tNeutron = tFactory->Create(2112);
    ASSERT_PTR(tNeutron);
    EXPECT_EQ(tNeutron->GetMass(), katrin::KConst::M_neut_kg());
    EXPECT_EQ(tNeutron->GetCharge(), 0.);
    delete tNeutron;

    // do not delete tFactory instance!
}
