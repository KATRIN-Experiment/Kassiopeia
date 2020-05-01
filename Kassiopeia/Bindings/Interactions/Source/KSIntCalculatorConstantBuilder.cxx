#include "KSIntCalculatorConstantBuilder.h"

#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

template<> KSIntCalculatorConstantBuilder::~KComplexElement() {}

STATICINT sKSIntCalculatorConstantStructure = KSIntCalculatorConstantBuilder::Attribute<string>("name") +
                                              KSIntCalculatorConstantBuilder::Attribute<double>("cross_section");

STATICINT sToolboxKSIntCalculatorConstant =
    KSRootBuilder::ComplexElement<KSIntCalculatorConstant>("ksint_calculator_constant");
}  // namespace katrin
