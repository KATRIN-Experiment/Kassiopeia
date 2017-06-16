#include "KSTermOutputBuilder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

    template< >
    KSTermOutputBuilder::~KComplexElement()
    {
    }

    STATICINT sKSTermOutputStructure =
            KSTermOutputBuilder::Attribute< string >( "name" ) +
            KSTermOutputBuilder::Attribute< double >( "min_value" ) +
            KSTermOutputBuilder::Attribute< double >( "max_value" ) +
            KSTermOutputBuilder::Attribute< string >( "group" ) +
            KSTermOutputBuilder::Attribute< string >( "component" );



    STATICINT sKSTermOutput =
            KSRootBuilder::ComplexElement< KSTermOutputData >( "ksterm_output" );

}
