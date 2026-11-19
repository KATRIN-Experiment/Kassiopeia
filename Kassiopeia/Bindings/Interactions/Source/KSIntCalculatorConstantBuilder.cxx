#include "KSIntCalculatorConstantBuilder.h"

#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

template<> KSIntCalculatorConstantBuilder::~KComplexElement() = default;

STATICINT sKSIntCalculatorConstantStructure =
    //! the name of the calculator.
    KSIntCalculatorConstantBuilder::Attribute<std::string>("name") +
    //! the elastic scattering cross section in meter^2.
    KSIntCalculatorConstantBuilder::Attribute<double>("cross_section");

STATICINT sToolboxKSIntCalculatorConstant =
    KSRootBuilder::ComplexElement<KSIntCalculatorConstant>("ksint_calculator_constant");
}  // namespace katrin
