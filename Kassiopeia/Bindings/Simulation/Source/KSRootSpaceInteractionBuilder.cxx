#include "KSRootSpaceInteractionBuilder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
namespace katrin
{

    template< >
    KSRootSpaceInteractionBuilder::~KComplexElement()
    {
    }

    static const int sKSRootSpaceInteraction =
        KSRootBuilder::ComplexElement< KSRootSpaceInteraction >( "ks_root_space_interaction" );

    static const int sKSRootSpaceInteractionStructure =
        KSRootSpaceInteractionBuilder::Attribute< string >( "name" ) +
        KSRootSpaceInteractionBuilder::Attribute< string >( "add_space_interaction" );

}
