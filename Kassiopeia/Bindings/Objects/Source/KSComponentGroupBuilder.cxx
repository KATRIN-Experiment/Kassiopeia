#include "KSComponentGroupBuilder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
namespace katrin
{

    template< >
    KSComponentGroupBuilder::~KComplexElement()
    {
    }

    static int sKSGroupStructure =
        KSComponentGroupBuilder::Attribute< string >( "name" ) +
        KSComponentGroupBuilder::Attribute< string >( "output" );

    static int sKSGroup =
        KSRootBuilder::ComplexElement< KSComponentGroup >( "ks_component_group" );

}
