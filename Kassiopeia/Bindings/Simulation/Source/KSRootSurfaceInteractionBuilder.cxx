#include "KSRootSurfaceInteractionBuilder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
namespace katrin
{

    template< >
    KSRootSurfaceInteractionBuilder::~KComplexElement()
    {
    }

    static int sKSRootSurfaceInteraction =
        KSRootBuilder::ComplexElement< KSRootSurfaceInteraction >( "ks_root_surface_interaction" );

    static int sKSRootSurfaceInteractionStructure =
        KSRootSurfaceInteractionBuilder::Attribute< string >( "name" ) +
        KSRootSurfaceInteractionBuilder::Attribute< string >( "set_surface_interaction" );

}
