#include "KSWriteROOTConditionStepBuilder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

    template< >
    KSWriteROOTConditionStepBuilder::~KComplexElement()
    {
    }

    STATICINT sKSWriteROOTConditionStepStructure =
            KSWriteROOTConditionStepBuilder::Attribute< string >( "name" ) +
            KSWriteROOTConditionStepBuilder::Attribute< int >( "nth_step" ) +
            KSWriteROOTConditionStepBuilder::Attribute< string >( "group" ) +
            KSWriteROOTConditionStepBuilder::Attribute< string >( "parent" );



    STATICINT sKSWriteROOTConditionStep =
            KSRootBuilder::ComplexElement< KSWriteROOTConditionStepData >( "kswrite_root_condition_step" );

}
