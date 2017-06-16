#include "KSRootSpaceNavigatorBuilder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

    template< >
    KSRootSpaceNavigatorBuilder::~KComplexElement()
    {
    }

    STATICINT sKSRootSpaceNavigator =
        KSRootBuilder::ComplexElement< KSRootSpaceNavigator >( "ks_root_space_navigator" );

    STATICINT sKSRootSpaceNavigatorStructure =
        KSRootSpaceNavigatorBuilder::Attribute< string >( "name" ) +
        KSRootSpaceNavigatorBuilder::Attribute< string >( "set_space_navigator" );

}
