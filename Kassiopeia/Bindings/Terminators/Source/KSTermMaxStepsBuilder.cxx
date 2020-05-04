#include "KSTermMaxStepsBuilder.h"

#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

template<> KSTermMaxStepsBuilder::~KComplexElement() {}

STATICINT sKSTermMaxStepsStructure =
    KSTermMaxStepsBuilder::Attribute<string>("name") + KSTermMaxStepsBuilder::Attribute<unsigned int>("steps");

STATICINT sKSTermMaxSteps = KSRootBuilder::ComplexElement<KSTermMaxSteps>("ksterm_max_steps");

}  // namespace katrin
