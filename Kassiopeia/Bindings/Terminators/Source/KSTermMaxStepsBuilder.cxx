#include "KSTermMaxStepsBuilder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
namespace katrin
{

    template< >
    KSTermMaxStepsBuilder::~KComplexElement()
    {
    }

    static int sKSTermMaxStepsStructure =
        KSTermMaxStepsBuilder::Attribute< string >( "name" ) +
        KSTermMaxStepsBuilder::Attribute< unsigned int >( "steps" );

    static int sKSTermMaxSteps =
        KSRootBuilder::ComplexElement< KSTermMaxSteps >( "ksterm_max_steps" );

}
