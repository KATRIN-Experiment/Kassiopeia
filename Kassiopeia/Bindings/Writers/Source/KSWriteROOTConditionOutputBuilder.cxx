#include "KSWriteROOTConditionOutputBuilder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

    template< >
    KSWriteROOTConditionOutputBuilder::~KComplexElement()
    {
    }

    STATICINT sKSWriteROOTConditionOutputStructure =
            KSWriteROOTConditionOutputBuilder::Attribute< string >( "name" ) +
            KSWriteROOTConditionOutputBuilder::Attribute< double >( "min_value" ) +
            KSWriteROOTConditionOutputBuilder::Attribute< double >( "max_value" ) +
            KSWriteROOTConditionOutputBuilder::Attribute< string >( "group" ) +
            KSWriteROOTConditionOutputBuilder::Attribute< string >( "parent" );



    STATICINT sKSWriteROOTConditionOutput =
            KSRootBuilder::ComplexElement< KSWriteROOTConditionOutputData >( "kswrite_root_condition_output" );

}
