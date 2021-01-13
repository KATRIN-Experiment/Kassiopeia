#include "KSIntCalculatorConstantBuilder.h"

#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

template<> KSIntCalculatorConstantBuilder::~KComplexElement() = default;

STATICINT sKSIntCalculatorConstantStructure = KSIntCalculatorConstantBuilder::Attribute<std::string>("name") +
                                              KSIntCalculatorConstantBuilder::Attribute<double>("cross_section");

STATICINT sToolboxKSIntCalculatorConstant =
    KSRootBuilder::ComplexElement<KSIntCalculatorConstant>("ksint_calculator_constant");
}  // namespace katrin
