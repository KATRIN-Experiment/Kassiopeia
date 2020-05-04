#ifndef Kassiopeia_KSTrajControlEnergyBuilder_h_
#define Kassiopeia_KSTrajControlEnergyBuilder_h_

#include "KComplexElement.hh"
#include "KSTrajControlEnergy.h"

using namespace Kassiopeia;
namespace katrin
{

typedef KComplexElement<KSTrajControlEnergy> KSTrajControlEnergyBuilder;

template<> inline bool KSTrajControlEnergyBuilder::AddAttribute(KContainer* aContainer)
{
    if (aContainer->GetName() == "name") {
        aContainer->CopyTo(fObject, &KNamed::SetName);
        return true;
    }
    if (aContainer->GetName() == "lower_limit") {
        aContainer->CopyTo(fObject, &KSTrajControlEnergy::SetLowerLimit);
        return true;
    }
    if (aContainer->GetName() == "upper_limit") {
        aContainer->CopyTo(fObject, &KSTrajControlEnergy::SetUpperLimit);
        return true;
    }
    if (aContainer->GetName() == "min_length") {
        aContainer->CopyTo(fObject, &KSTrajControlEnergy::SetMinLength);
        return true;
    }
    if (aContainer->GetName() == "max_length") {
        aContainer->CopyTo(fObject, &KSTrajControlEnergy::SetMaxLength);
        return true;
    }
    if (aContainer->GetName() == "initial_step") {
        aContainer->CopyTo(fObject, &KSTrajControlEnergy::SetInitialStep);
        return true;
    }
    if (aContainer->GetName() == "adjustment") {
        aContainer->CopyTo(fObject, &KSTrajControlEnergy::SetAdjustmentFactorUp);
        aContainer->CopyTo(fObject, &KSTrajControlEnergy::SetAdjustmentFactorDown);
        return true;
    }
    if (aContainer->GetName() == "adjustment_up") {
        aContainer->CopyTo(fObject, &KSTrajControlEnergy::SetAdjustmentFactorUp);
        return true;
    }
    if (aContainer->GetName() == "adjustment_down") {
        aContainer->CopyTo(fObject, &KSTrajControlEnergy::SetAdjustmentFactorDown);
        return true;
    }
    if (aContainer->GetName() == "step_rescale") {
        aContainer->CopyTo(fObject, &KSTrajControlEnergy::SetStepRescale);
        return true;
    }
    return false;
}

}  // namespace katrin
#endif
