#include "KSNavSpaceBuilder.h"

#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

template<> KSNavSpaceBuilder::~KComplexElement() = default;

STATICINT sKSNavSpaceStructure =
    KSNavSpaceBuilder::Attribute<std::string>("name") + KSNavSpaceBuilder::Attribute<bool>("enter_split") +
    KSNavSpaceBuilder::Attribute<bool>("exit_split") + KSNavSpaceBuilder::Attribute<bool>("fail_check") +
    KSNavSpaceBuilder::Attribute<double>("tolerance");

STATICINT sToolboxKSNavSpace = KSRootBuilder::ComplexElement<KSNavSpace>("ksnav_space");

}  // namespace katrin
