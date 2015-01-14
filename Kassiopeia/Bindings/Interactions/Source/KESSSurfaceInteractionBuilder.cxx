#include "KSRootBuilder.h"
#include "KESSSurfaceInteractionBuilder.h"

using namespace Kassiopeia;
namespace katrin
{

    template< >
    KESSSurfaceInteractionBuilder::~KComplexElement()
    {
    }

    static int sKESSSSurfaceInteractionStructure =
            KESSSurfaceInteractionBuilder::Attribute< string >( "name" ) +
            KESSSurfaceInteractionBuilder::Attribute< string >( "siliconside" );

    static int sKESSSSurfaceInteractionElement =
            KSRootBuilder::ComplexElement< KESSSurfaceInteraction >( "kess_surface_interaction" );
}
