#include "KSRootSpaceNavigatorBuilder.h"

#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

template<> KSRootSpaceNavigatorBuilder::~KComplexElement() = default;

STATICINT sKSRootSpaceNavigator = KSRootBuilder::ComplexElement<KSRootSpaceNavigator>("ks_root_space_navigator");

STATICINT sKSRootSpaceNavigatorStructure = KSRootSpaceNavigatorBuilder::Attribute<std::string>("name") +
                                           KSRootSpaceNavigatorBuilder::Attribute<std::string>("set_space_navigator");

}  // namespace katrin
