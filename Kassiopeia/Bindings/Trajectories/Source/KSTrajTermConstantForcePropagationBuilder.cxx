#include "KSRootBuilder.h"
#include "KSTrajTermConstantForcePropagationBuilder.h"


using namespace Kassiopeia;
using namespace std;

namespace katrin
{

    template< >
    KSTrajTermConstantForcePropagationBuilder::~KComplexElement()
    {
    }

    STATICINT sKSTrajTermConstantForcePropagationStructure =
        KSTrajTermConstantForcePropagationBuilder::Attribute< string >( "name" ) +
        KSTrajTermConstantForcePropagationBuilder::Attribute< KThreeVector >( "force" );

    STATICINT sToolboxKSTrajTermConstantForcePropagation =
        KSRootBuilder::ComplexElement< KSTrajTermConstantForcePropagation >( "kstraj_term_constant_force_propagation" );

}
