#include "KSIntCalculatorMottBuilder.h"

#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

template<> KSIntCalculatorMottBuilder::~KComplexElement() = default;

STATICINT sKSIntCalculatorMottStructure =
    KSIntCalculatorMottBuilder::Attribute<std::string>("name") +
    //! set the minimum angle (in degrees) allowable to scatter into. 0 not allowed, because singularity in cross section.
    KSIntCalculatorMottBuilder::Attribute<double>("theta_min") +
    //! set the maximum angle (in degrees) allowable to scatter into. 180 is the upper limit.
    KSIntCalculatorMottBuilder::Attribute<double>("theta_max") +
    //! set the nucleus name to scatter from either He-4 or Ne-20 nuclei.
    KSIntCalculatorMottBuilder::Attribute<std::string>("nucleus");

STATICINT sToolboxKSIntCalculatorMott =
    KSRootBuilder::ComplexElement<KSIntCalculatorMott>("ksint_calculator_mott");
}  // namespace katrin
