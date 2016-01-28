#include "KSRootSurfaceInteractionBuilder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
namespace katrin
{

    template< >
    KSRootSurfaceInteractionBuilder::~KComplexElement()
    {
    }

    STATICINT sKSRootSurfaceInteraction =
        KSRootBuilder::ComplexElement< KSRootSurfaceInteraction >( "ks_root_surface_interaction" );

    STATICINT sKSRootSurfaceInteractionStructure =
        KSRootSurfaceInteractionBuilder::Attribute< string >( "name" ) +
        KSRootSurfaceInteractionBuilder::Attribute< string >( "set_surface_interaction" );

}
