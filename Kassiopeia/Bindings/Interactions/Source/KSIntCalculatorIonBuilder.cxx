#include "KSIntCalculatorIonBuilder.h"

#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

template<> KSIntCalculatorIonBuilder::~KComplexElement() = default;

STATICINT sKSIntCalculatorIonStructure = KSIntCalculatorIonBuilder::Attribute<std::string>("name") +
                                         KSIntCalculatorIonBuilder::Attribute<std::string>("gas");

STATICINT sToolboxKSIntCalculatorIon = KSRootBuilder::ComplexElement<KSIntCalculatorIon>("ksint_calculator_ion");
}  // namespace katrin
