#ifndef Kassiopeia_KSGenValueFermiBuilder_h_
#define Kassiopeia_KSGenValueFermiBuilder_h_

#include "KComplexElement.hh"
#include "KSGenValueFermi.h"

using namespace Kassiopeia;
namespace katrin
{

typedef KComplexElement<KSGenValueFermi> KSGenValueFermiBuilder;

template<> inline bool KSGenValueFermiBuilder::AddAttribute(KContainer* aContainer)
{
    if (aContainer->GetName() == "name") {
        aContainer->CopyTo(fObject, &KNamed::SetName);
        return true;
    }
    if (aContainer->GetName() == "value_min") {
        aContainer->CopyTo(fObject, &KSGenValueFermi::SetValueMin);
        return true;
    }
    if (aContainer->GetName() == "value_max") {
        aContainer->CopyTo(fObject, &KSGenValueFermi::SetValueMax);
        return true;
    }
    if (aContainer->GetName() == "value_mean") {
        aContainer->CopyTo(fObject, &KSGenValueFermi::SetValueMean);
        return true;
    }
    if (aContainer->GetName() == "value_tau") {
        aContainer->CopyTo(fObject, &KSGenValueFermi::SetValueTau);
        return true;
    }
    if (aContainer->GetName() == "value_temp") {
        aContainer->CopyTo(fObject, &KSGenValueFermi::SetValueTemp);
        return true;
    }
    return false;
}

}  // namespace katrin

#endif
