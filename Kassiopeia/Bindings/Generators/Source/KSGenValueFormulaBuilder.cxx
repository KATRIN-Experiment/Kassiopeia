#include "KSGenValueFormulaBuilder.h"

#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

template<> KSGenValueFormulaBuilder::~KComplexElement() = default;

STATICINT sKSGenValueFormulaStructure =
    //! the name of this value.
    KSGenValueFormulaBuilder::Attribute<std::string>("name") +
    //! the minimum value to return.
    KSGenValueFormulaBuilder::Attribute<double>("value_min") +
    //! the maximum value to return.
    KSGenValueFormulaBuilder::Attribute<double>("value_max") +
    //! the underlying distribution from which to draw.
    KSGenValueFormulaBuilder::Attribute<std::string>("value_formula");

STATICINT sKSGenValueFormula = KSRootBuilder::ComplexElement<KSGenValueFormula>("ksgen_value_formula");

}  // namespace katrin
