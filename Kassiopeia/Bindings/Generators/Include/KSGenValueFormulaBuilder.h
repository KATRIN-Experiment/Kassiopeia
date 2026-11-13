#ifndef Kassiopeia_KSGenValueFormulaBuilder_h_
#define Kassiopeia_KSGenValueFormulaBuilder_h_

#include "KComplexElement.hh"
#include "KSGenValueFormula.h"

using namespace Kassiopeia;
namespace katrin
{

//! generates a single random value drawn from a portion of a user-defined distribution.
//! implemented using a root tf1.
//! @example <ksgen_value_formula name="value_formula" value_min="0." value_max="1." value_formula="x * x / 2."/>
typedef KComplexElement<KSGenValueFormula> KSGenValueFormulaBuilder;

template<> inline bool KSGenValueFormulaBuilder::AddAttribute(KContainer* aContainer)
{
    if (aContainer->GetName() == "name") {
        aContainer->CopyTo(fObject, &KNamed::SetName);
        return true;
    }
    if (aContainer->GetName() == "value_min") {
        aContainer->CopyTo(fObject, &KSGenValueFormula::SetValueMin);
        return true;
    }
    if (aContainer->GetName() == "value_max") {
        aContainer->CopyTo(fObject, &KSGenValueFormula::SetValueMax);
        return true;
    }
    if (aContainer->GetName() == "value_formula") {
        aContainer->CopyTo(fObject, &KSGenValueFormula::SetValueFormula);
        return true;
    }
    return false;
}

}  // namespace katrin

#endif
