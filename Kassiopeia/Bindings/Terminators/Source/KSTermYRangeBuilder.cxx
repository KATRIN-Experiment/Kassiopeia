#include "KSTermYRangeBuilder.h"

#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

template<> KSTermYRangeBuilder::~KComplexElement() = default;

STATICINT sKSTermYRangeStructure =
    KSTermYRangeBuilder::Attribute<std::string>("name") + 
    KSTermYRangeBuilder::Attribute<double>("ymin") +
    KSTermYRangeBuilder::Attribute<double>("ymax");

STATICINT sToolboxKSTermYRange = KSRootBuilder::ComplexElement<KSTermYRange>("ksterm_yrange");


}  // namespace katrin
