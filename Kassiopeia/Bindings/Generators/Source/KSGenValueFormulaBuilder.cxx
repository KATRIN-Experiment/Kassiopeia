#include "KSGenValueFormulaBuilder.h"

#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

template<> KSGenValueFormulaBuilder::~KComplexElement() {}

STATICINT sKSGenValueFormulaStructure = KSGenValueFormulaBuilder::Attribute<string>("name") +
                                        KSGenValueFormulaBuilder::Attribute<double>("value_min") +
                                        KSGenValueFormulaBuilder::Attribute<double>("value_max") +
                                        KSGenValueFormulaBuilder::Attribute<string>("value_formula");

STATICINT sKSGenValueFormula = KSRootBuilder::ComplexElement<KSGenValueFormula>("ksgen_value_formula");

}  // namespace katrin
