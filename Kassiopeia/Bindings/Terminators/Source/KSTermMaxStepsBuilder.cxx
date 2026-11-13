#include "KSTermMaxStepsBuilder.h"

#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

template<> KSTermMaxStepsBuilder::~KComplexElement() = default;

STATICINT sKSTermMaxStepsStructure =
    //! the name of the terminator.
    KSTermMaxStepsBuilder::Attribute<std::string>("name") +
    //! the maximum allowed track step count.
    KSTermMaxStepsBuilder::Attribute<unsigned int>("steps");

STATICINT sKSTermMaxSteps = KSRootBuilder::ComplexElement<KSTermMaxSteps>("ksterm_max_steps");

}  // namespace katrin
