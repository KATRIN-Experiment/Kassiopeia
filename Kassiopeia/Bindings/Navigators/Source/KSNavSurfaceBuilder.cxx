#include "KSNavSurfaceBuilder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
namespace katrin
{

    template< >
    KSNavSurfaceBuilder::~KComplexElement()
    {
    }

    static int sKSNavSurfaceStructure =
        KSNavSurfaceBuilder::Attribute< string >( "name" ) +
        KSNavSurfaceBuilder::Attribute< bool >( "transmission_split" ) +
        KSNavSurfaceBuilder::Attribute< bool >( "reflection_split" );

    static int sToolboxKSNavSurface =
        KSRootBuilder::ComplexElement< KSNavSurface >( "ksnav_surface" );


}
