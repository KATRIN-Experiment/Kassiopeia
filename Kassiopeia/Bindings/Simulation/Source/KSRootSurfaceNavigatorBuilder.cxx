#include "KSRootSurfaceNavigatorBuilder.h"

#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

template<> KSRootSurfaceNavigatorBuilder::~KComplexElement() = default;

STATICINT sKSRootSurfaceNavigator = KSRootBuilder::ComplexElement<KSRootSurfaceNavigator>("ks_root_surface_navigator");

STATICINT sKSRootSurfaceNavigatorStructure =
    KSRootSurfaceNavigatorBuilder::Attribute<std::string>("name") +
    KSRootSurfaceNavigatorBuilder::Attribute<std::string>("set_surface_navigator");

}  // namespace katrin
