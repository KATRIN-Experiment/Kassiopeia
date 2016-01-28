#include "KSRootBuilder.h"
#include "KESSSurfaceInteractionBuilder.h"

using namespace Kassiopeia;
namespace katrin
{

    template< >
    KESSSurfaceInteractionBuilder::~KComplexElement()
    {
    }

    STATICINT sKESSSSurfaceInteractionStructure =
            KESSSurfaceInteractionBuilder::Attribute< string >( "name" ) +
            KESSSurfaceInteractionBuilder::Attribute< string >( "siliconside" );

    STATICINT sKESSSSurfaceInteractionElement =
            KSRootBuilder::ComplexElement< KESSSurfaceInteraction >( "kess_surface_interaction" );
}
