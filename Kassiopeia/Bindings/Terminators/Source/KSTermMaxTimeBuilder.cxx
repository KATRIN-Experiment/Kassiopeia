#include "KSTermMaxTimeBuilder.h"

#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

template<> KSTermMaxTimeBuilder::~KComplexElement() {}

STATICINT sKSTermMaxTimeStructure =
    KSTermMaxTimeBuilder::Attribute<string>("name") + KSTermMaxTimeBuilder::Attribute<double>("time");

STATICINT sKSTermMaxTime = KSRootBuilder::ComplexElement<KSTermMaxTime>("ksterm_max_time");

}  // namespace katrin
