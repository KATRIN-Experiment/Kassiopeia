#ifndef Kassiopeia_KSIntCalculatorMottBuilder_h_
#define Kassiopeia_KSIntCalculatorMottBuilder_h_

#include "KComplexElement.hh"
#include "KSIntCalculatorMott.h"

using namespace Kassiopeia;
namespace katrin
{

typedef KComplexElement<KSIntCalculatorMott> KSIntCalculatorMottBuilder;

template<> inline bool KSIntCalculatorMottBuilder::AddAttribute(KContainer* aContainer)
{
    if (aContainer->GetName() == "name") {
        aContainer->CopyTo(fObject, &KNamed::SetName);
        return true;
    }
    if (aContainer->GetName() == "theta_min") {
        aContainer->CopyTo(fObject, &KSIntCalculatorMott::SetThetaMin);
        return true;
    }
    if (aContainer->GetName() == "theta_max") {
        aContainer->CopyTo(fObject, &KSIntCalculatorMott::SetThetaMax);
        return true;
    }
    return false;
}

}  // namespace katrin

#endif
