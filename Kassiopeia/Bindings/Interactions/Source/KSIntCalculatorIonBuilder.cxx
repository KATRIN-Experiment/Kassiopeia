#include "KSIntCalculatorIonBuilder.h"

#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

template<> KSIntCalculatorIonBuilder::~KComplexElement() {}

STATICINT sKSIntCalculatorIonStructure =
    KSIntCalculatorIonBuilder::Attribute<string>("name") + KSIntCalculatorIonBuilder::Attribute<string>("gas");

STATICINT sToolboxKSIntCalculatorIon = KSRootBuilder::ComplexElement<KSIntCalculatorIon>("ksint_calculator_ion");
}  // namespace katrin
