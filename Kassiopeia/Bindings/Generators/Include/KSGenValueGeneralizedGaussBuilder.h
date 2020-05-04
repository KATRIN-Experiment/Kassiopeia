#ifndef Kassiopeia_KSGenValueGeneralizedGaussBuilder_h_
#define Kassiopeia_KSGenValueGeneralizedGaussBuilder_h_

#include "KComplexElement.hh"
#include "KSGenValueGeneralizedGauss.h"

using namespace Kassiopeia;
namespace katrin
{

typedef KComplexElement<KSGenValueGeneralizedGauss> KSGenValueGeneralizedGaussBuilder;

template<> inline bool KSGenValueGeneralizedGaussBuilder::AddAttribute(KContainer* aContainer)
{
    if (aContainer->GetName() == "name") {
        aContainer->CopyTo(fObject, &KNamed::SetName);
        return true;
    }
    if (aContainer->GetName() == "value_min") {
        aContainer->CopyTo(fObject, &KSGenValueGeneralizedGauss::SetValueMin);
        return true;
    }
    if (aContainer->GetName() == "value_max") {
        aContainer->CopyTo(fObject, &KSGenValueGeneralizedGauss::SetValueMax);
        return true;
    }
    if (aContainer->GetName() == "value_mean") {
        aContainer->CopyTo(fObject, &KSGenValueGeneralizedGauss::SetValueMean);
        return true;
    }
    if (aContainer->GetName() == "value_sigma") {
        aContainer->CopyTo(fObject, &KSGenValueGeneralizedGauss::SetValueSigma);
        return true;
    }
    if (aContainer->GetName() == "value_skew") {
        aContainer->CopyTo(fObject, &KSGenValueGeneralizedGauss::SetValueSkew);
        return true;
    }
    return false;
}

}  // namespace katrin

#endif
