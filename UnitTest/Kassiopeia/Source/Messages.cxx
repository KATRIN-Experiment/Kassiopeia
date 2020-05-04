/**
 * Unit testing for Kassiopeia's message definitions
 * @author J. Behrens
 *
 * This file contains a unit tests for most of Kassiopeia's messages.
 * All tests should be grouped together in a meaningful way, and should use
 * a fixture class which is derived from TimeoutTest. Please keep unit tests
 * as simple as possible (think of a minimal example of how to use a certain
 * class). It is a good idea to also include a death-test for each class.
 *
 * See the official GoogleTest pages for more info:
 *   https://code.google.com/p/googletest/wiki/Primer
 *   https://code.google.com/p/googletest/wiki/AdvancedGuide
 */

#include "KSEventMessage.h"  // from Simulation
#include "KSFieldsMessage.h"
#include "KSGeneratorsMessage.h"
#include "KSInteractionsMessage.h"
#include "KSMathMessage.h"
#include "KSNavigatorsMessage.h"
#include "KSObjectsMessage.h"
#include "KSOperatorsMessage.h"
#include "KSReadersMessage.h"
#include "KSRunMessage.h"   // from Simulation
#include "KSStepMessage.h"  // from Simulation
#include "KSTerminatorsMessage.h"
#include "KSTrackMessage.h"  // from Simulation
#include "KSTrajectoriesMessage.h"
#include "KSUtilityMessage.h"
#include "KSVisualizationMessage.h"
#include "KSWritersMessage.h"
#include "UnitTest.h"

using namespace Kassiopeia;


/////////////////////////////////////////////////////////////////////////////
// Messages Unit Testing
/////////////////////////////////////////////////////////////////////////////

TEST(KassiopeiaMessageTest, KSFieldsMessage)
{
    auto* tMessageClass = new Kassiopeia::KMessage_fieldmsg();
    EXPECT_PTR(tMessageClass);
    delete tMessageClass;
}

TEST(KassiopeiaMessageTest, KSGeneratorsMessage)
{
    auto* tMessageClass = new Kassiopeia::KMessage_genmsg();
    EXPECT_PTR(tMessageClass);
    delete tMessageClass;
}

TEST(KassiopeiaMessageTest, KSInteractionsMessage)
{
    auto* tMessageClass = new Kassiopeia::KMessage_intmsg();
    EXPECT_PTR(tMessageClass);
    delete tMessageClass;
}

TEST(KassiopeiaMessageTest, KSMathMessage)
{
    auto* tMessageClass = new Kassiopeia::KMessage_mathmsg();
    EXPECT_PTR(tMessageClass);
    delete tMessageClass;
}

TEST(KassiopeiaMessageTest, KSNavigatorsMessage)
{
    auto* tMessageClass = new Kassiopeia::KMessage_navmsg();
    EXPECT_PTR(tMessageClass);
    delete tMessageClass;
}

TEST(KassiopeiaMessageTest, KSObjectsMessage)
{
    auto* tMessageClass = new Kassiopeia::KMessage_objctmsg();
    EXPECT_PTR(tMessageClass);
    delete tMessageClass;
}

TEST(KassiopeiaMessageTest, KSOperatorsMessage)
{
    auto* tMessageClass = new Kassiopeia::KMessage_oprmsg();
    EXPECT_PTR(tMessageClass);
    delete tMessageClass;
}

TEST(KassiopeiaMessageTest, KSReadersMessage)
{
    auto* tMessageClass = new Kassiopeia::KMessage_readermsg();
    EXPECT_PTR(tMessageClass);
    delete tMessageClass;
}

TEST(KassiopeiaMessageTest, KSRunMessage)
{
    auto* tMessageClass = new Kassiopeia::KMessage_runmsg();
    EXPECT_PTR(tMessageClass);
    delete tMessageClass;
}

TEST(KassiopeiaMessageTest, KSEventMessage)
{
    auto* tMessageClass = new Kassiopeia::KMessage_eventmsg();
    EXPECT_PTR(tMessageClass);
    delete tMessageClass;
}

TEST(KassiopeiaMessageTest, KSTrackMessage)
{
    auto* tMessageClass = new Kassiopeia::KMessage_trackmsg();
    EXPECT_PTR(tMessageClass);
    delete tMessageClass;
}

TEST(KassiopeiaMessageTest, KSStepMessage)
{
    auto* tMessageClass = new Kassiopeia::KMessage_stepmsg();
    EXPECT_PTR(tMessageClass);
    delete tMessageClass;
}

TEST(KassiopeiaMessageTest, KSTerminatorsMessage)
{
    auto* tMessageClass = new Kassiopeia::KMessage_termmsg();
    EXPECT_PTR(tMessageClass);
    delete tMessageClass;
}

TEST(KassiopeiaMessageTest, KSTrajectoriesMessage)
{
    auto* tMessageClass = new Kassiopeia::KMessage_trajmsg();
    EXPECT_PTR(tMessageClass);
    delete tMessageClass;
}

TEST(KassiopeiaMessageTest, KSUtilityMessage)
{
    auto* tMessageClass = new Kassiopeia::KMessage_ksutilmsg();
    EXPECT_PTR(tMessageClass);
    delete tMessageClass;
}

TEST(KassiopeiaMessageTest, KSVisualizationMessage)
{
    auto* tMessageClass = new Kassiopeia::KMessage_vismsg();
    EXPECT_PTR(tMessageClass);
    delete tMessageClass;
}

TEST(KassiopeiaMessageTest, KSWritersMessage)
{
    auto* tMessageClass = new Kassiopeia::KMessage_wtrmsg();
    EXPECT_PTR(tMessageClass);
    delete tMessageClass;
}
