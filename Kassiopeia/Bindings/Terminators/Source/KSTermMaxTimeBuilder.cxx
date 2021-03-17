#include "KSTermMaxTimeBuilder.h"

#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

template<> KSTermMaxTimeBuilder::~KComplexElement() = default;

STATICINT sKSTermMaxTimeStructure =
    KSTermMaxTimeBuilder::Attribute<std::string>("name") + KSTermMaxTimeBuilder::Attribute<double>("time");

STATICINT sKSTermMaxTime = KSRootBuilder::ComplexElement<KSTermMaxTime>("ksterm_max_time");

}  // namespace katrin
