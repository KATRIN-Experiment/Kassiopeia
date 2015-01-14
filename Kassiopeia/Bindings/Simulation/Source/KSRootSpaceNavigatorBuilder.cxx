#include "KSRootSpaceNavigatorBuilder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
namespace katrin
{

    template< >
    KSRootSpaceNavigatorBuilder::~KComplexElement()
    {
    }

    static const int sKSRootSpaceNavigator =
        KSRootBuilder::ComplexElement< KSRootSpaceNavigator >( "ks_root_space_navigator" );

    static const int sKSRootSpaceNavigatorStructure =
        KSRootSpaceNavigatorBuilder::Attribute< string >( "name" ) +
        KSRootSpaceNavigatorBuilder::Attribute< string >( "set_space_navigator" );

}
