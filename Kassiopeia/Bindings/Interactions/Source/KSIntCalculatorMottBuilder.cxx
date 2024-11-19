#include "KSIntCalculatorMottBuilder.h"

#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

template<> KSIntCalculatorMottBuilder::~KComplexElement() = default;

STATICINT sKSIntCalculatorMottStructure = KSIntCalculatorMottBuilder::Attribute<std::string>("name") +
                                              KSIntCalculatorMottBuilder::Attribute<double>("theta_min") +
                                              KSIntCalculatorMottBuilder::Attribute<double>("theta_max") +
                                              KSIntCalculatorMottBuilder::Attribute<std::string>("nucleus");

STATICINT sToolboxKSIntCalculatorMott =
    KSRootBuilder::ComplexElement<KSIntCalculatorMott>("calculator_mott");
}  // namespace katrin
