#include "KSGenValueFormulaBuilder.h"

#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

template<> KSGenValueFormulaBuilder::~KComplexElement() = default;

STATICINT sKSGenValueFormulaStructure = KSGenValueFormulaBuilder::Attribute<std::string>("name") +
                                        KSGenValueFormulaBuilder::Attribute<double>("value_min") +
                                        KSGenValueFormulaBuilder::Attribute<double>("value_max") +
                                        KSGenValueFormulaBuilder::Attribute<std::string>("value_formula");

STATICINT sKSGenValueFormula = KSRootBuilder::ComplexElement<KSGenValueFormula>("ksgen_value_formula");

}  // namespace katrin
