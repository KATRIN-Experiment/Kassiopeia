#ifndef Kassiopeia_KSGenValueGaussBuilder_h_
#define Kassiopeia_KSGenValueGaussBuilder_h_

#include "KComplexElement.hh"
#include "KSGenValueGauss.h"

using namespace Kassiopeia;
namespace katrin
{

//! generates a single random value drawn from a portion of a gaussian distribution.
//! @example <ksgen_value_gauss name="value_gauss" value_min="-10." value_max="10." value_mean="0." value_sigma="1."/>
typedef KComplexElement<KSGenValueGauss> KSGenValueGaussBuilder;

template<> inline bool KSGenValueGaussBuilder::AddAttribute(KContainer* aContainer)
{
    if (aContainer->GetName() == "name") {
        aContainer->CopyTo(fObject, &KNamed::SetName);
        return true;
    }
    if (aContainer->GetName() == "value_min") {
        aContainer->CopyTo(fObject, &KSGenValueGauss::SetValueMin);
        return true;
    }
    if (aContainer->GetName() == "value_max") {
        aContainer->CopyTo(fObject, &KSGenValueGauss::SetValueMax);
        return true;
    }
    if (aContainer->GetName() == "value_mean") {
        aContainer->CopyTo(fObject, &KSGenValueGauss::SetValueMean);
        return true;
    }
    if (aContainer->GetName() == "value_sigma") {
        aContainer->CopyTo(fObject, &KSGenValueGauss::SetValueSigma);
        return true;
    }
    return false;
}

}  // namespace katrin

#endif
