#include "KSComponentMemberBuilder.h"
#include "KSComponentGroupBuilder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
namespace katrin
{

    template< >
    KSComponentBuilder::~KComplexElement()
    {
    }

    static int sKSComponentStructure =
        KSComponentBuilder::Attribute< string >( "name" ) +
        KSComponentBuilder::Attribute< string >( "parent" ) +
        KSComponentBuilder::Attribute< string >( "field" );

    static int sKSComponent =
        KSComponentGroupBuilder::ComplexElement< KSComponentMemberData >( "component_member" ) +
        KSRootBuilder::ComplexElement< KSComponentMemberData >( "ks_component_member" );

}
