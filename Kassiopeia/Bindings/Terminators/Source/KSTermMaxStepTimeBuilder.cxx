#include "KSTermMaxStepTimeBuilder.h"

#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

template<> KSTermMaxStepTimeBuilder::~KComplexElement() {}

STATICINT sKSTermMaxStepTimeStructure =
    KSTermMaxStepTimeBuilder::Attribute<string>("name") + KSTermMaxStepTimeBuilder::Attribute<double>("time");

STATICINT sKSTermMaxStepTime = KSRootBuilder::ComplexElement<KSTermMaxStepTime>("ksterm_max_step_time");

}  // namespace katrin
