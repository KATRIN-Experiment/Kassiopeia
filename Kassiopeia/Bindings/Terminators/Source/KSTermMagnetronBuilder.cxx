#include "KSTermMagnetronBuilder.h"

#include "KSRootBuilder.h"

using namespace Kassiopeia;
namespace katrin
{

template<> KSTermMagnetronBuilder::~KComplexElement() = default;

STATICINT sKSTermMagnetronStructure =
    KSTermMagnetronBuilder::Attribute<std::string>("name") + KSTermMagnetronBuilder::Attribute<double>("max_phi");

STATICINT sKSTermMagnetron = KSRootBuilder::ComplexElement<KSTermMagnetron>("ksterm_magnetron");

}  // namespace katrin
