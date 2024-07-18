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

        if ((fObject->GetThetaMin() <= 0.0)) {
            initmsg(eError) << "\"" << fObject->GetThetaMin()
                           << R"(" is an invalid lower bound for Mott scattering! Change to a value > 0)" << eom;

            return false;
        }
        else {
            return true;
        }
    }
    if (aContainer->GetName() == "theta_max") {
        aContainer->CopyTo(fObject, &KSIntCalculatorMott::SetThetaMax);
        if ((fObject->GetThetaMax() > 180.0)) {
            initmsg(eError) << "\"" << fObject->GetThetaMax()
                           << R"(" is an invalid upper bound for Mott scattering! Change to a value < 180)" << eom;

            return false;
        }
        else {
            return true;
        }
    }
    if (aContainer->GetName() == "nucleus") {
        aContainer->CopyTo(fObject, &KSIntCalculatorMott::SetNucleus);

        if ((fObject->GetNucleus().compare("He") != 0) && (fObject->GetNucleus().compare("Ne") != 0)) {
            initmsg(eError) << "\"" << fObject->GetNucleus()
                           << R"(" is not available for Mott scattering! Available gases: "He", "Ne")" << eom;

            return false;
        }
        else {
            return true;
        }
    }
    return false;
}

}  // namespace katrin

#endif
