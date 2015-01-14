#include "KSRootBuilder.h"
#include "KSTrajTermConstantForcePropagationBuilder.h"


using namespace Kassiopeia;
namespace katrin
{

    template< >
    KSTrajTermConstantForcePropagationBuilder::~KComplexElement()
    {
    }

    static int sKSTrajTermConstantForcePropagationStructure =
        KSTrajTermConstantForcePropagationBuilder::Attribute< string >( "name" ) +
        KSTrajTermConstantForcePropagationBuilder::Attribute< KThreeVector >( "force" );

    static int sToolboxKSTrajTermConstantForcePropagation =
        KSRootBuilder::ComplexElement< KSTrajTermConstantForcePropagation >( "kstraj_term_constant_force_propagation" );

}
