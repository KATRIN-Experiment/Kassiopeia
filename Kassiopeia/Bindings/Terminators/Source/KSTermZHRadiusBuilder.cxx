#include "KSTermZHRadiusBuilder.h"

#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

template<> KSTermZHRadiusBuilder::~KComplexElement() = default;

STATICINT sKSTermZHRadiusStructure =
    KSTermZHRadiusBuilder::Attribute<std::string>("name") +
    KSTermZHRadiusBuilder::Attribute<std::string>("magnetic_field") +
    KSTermZHRadiusBuilder::Attribute<std::string>("electric_field") +
    KSTermZHRadiusBuilder::Attribute<bool>("central_expansion") +
    KSTermZHRadiusBuilder::Attribute<bool>("remote_expansion");

STATICINT sKSTermZHRadius = KSRootBuilder::ComplexElement<KSTermZHRadius>("ksterm_zh_radius");

}  // namespace katrin
