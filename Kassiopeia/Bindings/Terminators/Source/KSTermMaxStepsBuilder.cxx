#include "KSTermMaxStepsBuilder.h"

#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

template<> KSTermMaxStepsBuilder::~KComplexElement() = default;

STATICINT sKSTermMaxStepsStructure =
    KSTermMaxStepsBuilder::Attribute<std::string>("name") + KSTermMaxStepsBuilder::Attribute<unsigned int>("steps");

STATICINT sKSTermMaxSteps = KSRootBuilder::ComplexElement<KSTermMaxSteps>("ksterm_max_steps");

}  // namespace katrin
