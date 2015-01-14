#include "KSRootSurfaceNavigatorBuilder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
namespace katrin
{

    template< >
    KSRootSurfaceNavigatorBuilder::~KComplexElement()
    {
    }

    static const int sKSRootSurfaceNavigator =
        KSRootBuilder::ComplexElement< KSRootSurfaceNavigator >( "ks_root_surface_navigator" );

    static const int sKSRootSurfaceNavigatorStructure =
        KSRootSurfaceNavigatorBuilder::Attribute< string >( "name" ) +
        KSRootSurfaceNavigatorBuilder::Attribute< string >( "set_surface_navigator" );

}
