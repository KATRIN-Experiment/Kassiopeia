#include "KSTermXRangeBuilder.h"

#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

template<> KSTermXRangeBuilder::~KComplexElement() = default;

STATICINT sKSTermXRangeStructure =
    KSTermXRangeBuilder::Attribute<std::string>("name") +
    KSTermXRangeBuilder::Attribute<double>("xmin") +
    KSTermXRangeBuilder::Attribute<double>("xmax");

STATICINT sToolboxKSTermXRange = KSRootBuilder::ComplexElement<KSTermXRange>("ksterm_xrange");


}  // namespace katrin
