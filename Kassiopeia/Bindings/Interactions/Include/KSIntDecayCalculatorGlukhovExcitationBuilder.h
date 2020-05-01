#ifndef Kassiopeia_KSIntDecayCalculatorGlukhovExcitationBuilder_h_
#define Kassiopeia_KSIntDecayCalculatorGlukhovExcitationBuilder_h_

#include "KComplexElement.hh"
#include "KSIntDecayCalculatorGlukhovExcitation.h"
#include "KToolbox.h"


using namespace Kassiopeia;
namespace katrin
{

typedef KComplexElement<KSIntDecayCalculatorGlukhovExcitation> KSIntDecayCalculatorGlukhovExcitationBuilder;

template<> inline bool KSIntDecayCalculatorGlukhovExcitationBuilder::AddAttribute(KContainer* aContainer)
{
    if (aContainer->GetName() == "name") {
        aContainer->CopyTo(fObject, &KNamed::SetName);
        return true;
    }
    if (aContainer->GetName() == "target_pid") {
        aContainer->CopyTo(fObject, &KSIntDecayCalculatorGlukhovExcitation::SetTargetPID);
        return true;
    }
    if (aContainer->GetName() == "min_pid") {
        aContainer->CopyTo(fObject, &KSIntDecayCalculatorGlukhovExcitation::SetminPID);
        return true;
    }
    if (aContainer->GetName() == "max_pid") {
        aContainer->CopyTo(fObject, &KSIntDecayCalculatorGlukhovExcitation::SetminPID);
        return true;
    }
    if (aContainer->GetName() == "temperature") {
        aContainer->CopyTo(fObject, &KSIntDecayCalculatorGlukhovExcitation::SetTemperature);
        return true;
    }

    return false;
}

}  // namespace katrin

#endif
