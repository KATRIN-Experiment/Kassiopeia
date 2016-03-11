#include "KSRootSurfaceNavigatorBuilder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
namespace katrin
{

    template< >
    KSRootSurfaceNavigatorBuilder::~KComplexElement()
    {
    }

    STATICINT sKSRootSurfaceNavigator =
        KSRootBuilder::ComplexElement< KSRootSurfaceNavigator >( "ks_root_surface_navigator" );

    STATICINT sKSRootSurfaceNavigatorStructure =
        KSRootSurfaceNavigatorBuilder::Attribute< string >( "name" ) +
        KSRootSurfaceNavigatorBuilder::Attribute< string >( "set_surface_navigator" );

}
