#include "KSIntCalculatorIonBuilder.h"

#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

template<> KSIntCalculatorIonBuilder::~KComplexElement() = default;

STATICINT sKSIntCalculatorIonStructure = KSIntCalculatorIonBuilder::Attribute<std::string>("name") +
                                         //! set the name of the gas species to be used.
                                         //! only H_2 is available at present and is used by default.
                                         KSIntCalculatorIonBuilder::Attribute<std::string>("gas");

STATICINT sToolboxKSIntCalculatorIon = KSRootBuilder::ComplexElement<KSIntCalculatorIon>("ksint_calculator_ion");
}  // namespace katrin
