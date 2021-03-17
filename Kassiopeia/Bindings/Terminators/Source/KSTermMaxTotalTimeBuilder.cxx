#include "KSTermMaxTotalTimeBuilder.h"

#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

template<> KSTermMaxTotalTimeBuilder::~KComplexElement() = default;

STATICINT sKSTermMaxTotalTimeStructure =
    KSTermMaxTotalTimeBuilder::Attribute<std::string>("name") + KSTermMaxTotalTimeBuilder::Attribute<double>("time");

STATICINT sKSTermMaxStepTime = KSRootBuilder::ComplexElement<KSTermMaxTotalTime>("ksterm_max_total_time");

}  // namespace katrin
