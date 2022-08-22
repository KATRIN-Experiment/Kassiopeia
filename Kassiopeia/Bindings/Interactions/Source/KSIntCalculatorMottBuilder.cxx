#include "KSIntCalculatorMottBuilder.h"

#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

template<> KSIntCalculatorMottBuilder::~KComplexElement() = default;

STATICINT sKSIntCalculatorMottStructure = KSIntCalculatorMottBuilder::Attribute<std::string>("name") +
                                              KSIntCalculatorMottBuilder::Attribute<double>("theta_min") +
                                              KSIntCalculatorMottBuilder::Attribute<double>("theta_max");

STATICINT sToolboxKSIntCalculatorMott =
    KSRootBuilder::ComplexElement<KSIntCalculatorMott>("ksint_calculator_mott");
}  // namespace katrin
