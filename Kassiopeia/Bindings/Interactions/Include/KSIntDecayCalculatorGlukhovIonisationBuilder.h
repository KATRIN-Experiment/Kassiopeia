#ifndef Kassiopeia_KSIntDecayCalculatorGlukhovIonisationBuilder_h_
#define Kassiopeia_KSIntDecayCalculatorGlukhovIonisationBuilder_h_

#include "KComplexElement.hh"
#include "KSIntDecayCalculatorGlukhovIonisation.h"
#include "KToolbox.h"


using namespace Kassiopeia;
namespace katrin
{

typedef KComplexElement<KSIntDecayCalculatorGlukhovIonisation> KSIntDecayCalculatorGlukhovIonisationBuilder;

template<> inline bool KSIntDecayCalculatorGlukhovIonisationBuilder::AddAttribute(KContainer* aContainer)
{
    if (aContainer->GetName() == "name") {
        aContainer->CopyTo(fObject, &KNamed::SetName);
        return true;
    }
    if (aContainer->GetName() == "target_pid") {
        aContainer->CopyTo(fObject, &KSIntDecayCalculatorGlukhovIonisation::SetTargetPID);
        return true;
    }
    if (aContainer->GetName() == "min_pid") {
        aContainer->CopyTo(fObject, &KSIntDecayCalculatorGlukhovIonisation::SetminPID);
        return true;
    }
    if (aContainer->GetName() == "max_pid") {
        aContainer->CopyTo(fObject, &KSIntDecayCalculatorGlukhovIonisation::SetminPID);
        return true;
    }
    if (aContainer->GetName() == "temperature") {
        aContainer->CopyTo(fObject, &KSIntDecayCalculatorGlukhovIonisation::SetTemperature);
        return true;
    }

    return false;
}

template<> inline bool KSIntDecayCalculatorGlukhovIonisationBuilder::AddElement(KContainer* aContainer)
{
    if (aContainer->Is<KSGenerator>() == true) {
        aContainer->ReleaseTo(fObject, &KSIntDecayCalculatorGlukhovIonisation::SetDecayProductGenerator);
        return true;
    }
    return false;
}

}  // namespace katrin

#endif
