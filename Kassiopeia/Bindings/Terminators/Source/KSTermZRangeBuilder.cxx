#include "KSTermZRangeBuilder.h"

#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

template<> KSTermZRangeBuilder::~KComplexElement() = default;

STATICINT sKSTermZRangeStructure =
    KSTermZRangeBuilder::Attribute<std::string>("name") + 
    KSTermZRangeBuilder::Attribute<double>("zmin") +
    KSTermZRangeBuilder::Attribute<double>("zmax");

STATICINT sToolboxKSTermZRange = KSRootBuilder::ComplexElement<KSTermZRange>("ksterm_zrange");


}  // namespace katrin
