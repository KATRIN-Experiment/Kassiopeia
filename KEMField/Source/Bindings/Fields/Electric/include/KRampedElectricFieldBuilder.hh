/*
 * KRampedElectricFieldBuilder.hh
 *
 *  Created on: 31 May 2016
 *      Author: wolfgang
 */

#ifndef KRAMPEDELECTRICFIELDBUILDER_HH_
#define KRAMPEDELECTRICFIELDBUILDER_HH_

#include "KComplexElement.hh"
#include "KRampedElectricField.hh"
#include "KToolbox.h"

namespace katrin
{

typedef KComplexElement<KEMField::KRampedElectricField> KRampedElectricFieldBuilder;

template<> inline bool KRampedElectricFieldBuilder::AddAttribute(KContainer* aContainer)
{
    if (aContainer->GetName() == "name") {
        std::string name;
        aContainer->CopyTo(name);
        this->SetName(name);
        fObject->SetName(name);
        return true;
    }
    if (aContainer->GetName() == "root_field") {
        std::string fieldName = aContainer->AsReference<std::string>();
        auto* field = katrin::KToolbox::GetInstance().Get<KEMField::KElectricField>(fieldName);
        fObject->SetRootElectricField(field);
        return true;
    }
    if (aContainer->GetName() == "ramping_type") {
        std::string tFlag = aContainer->AsReference<std::string>();
        if (tFlag == std::string("linear"))
            fObject->SetRampingType(KEMField::KRampedElectricField::rtLinear);
        else if (tFlag == std::string("exponential"))
            fObject->SetRampingType(KEMField::KRampedElectricField::rtExponential);
        else if (tFlag == std::string("sinus"))
            fObject->SetRampingType(KEMField::KRampedElectricField::rtSinus);
        return true;
    }
    if (aContainer->GetName() == "num_cycles") {
        aContainer->CopyTo(fObject, &KEMField::KRampedElectricField::SetNumCycles);
        return true;
    }
    if (aContainer->GetName() == "ramp_up_delay") {
        aContainer->CopyTo(fObject, &KEMField::KRampedElectricField::SetRampUpDelay);
        return true;
    }
    if (aContainer->GetName() == "ramp_down_delay") {
        aContainer->CopyTo(fObject, &KEMField::KRampedElectricField::SetRampDownDelay);
        return true;
    }
    if (aContainer->GetName() == "ramp_up_time") {
        aContainer->CopyTo(fObject, &KEMField::KRampedElectricField::SetRampUpTime);
        return true;
    }
    if (aContainer->GetName() == "ramp_down_time") {
        aContainer->CopyTo(fObject, &KEMField::KRampedElectricField::SetRampDownTime);
        return true;
    }
    if (aContainer->GetName() == "time_constant") {
        aContainer->CopyTo(fObject, &KEMField::KRampedElectricField::SetTimeConstant);
        return true;
    }
    if (aContainer->GetName() == "time_scaling") {
        aContainer->CopyTo(fObject, &KEMField::KRampedElectricField::SetTimeScalingFactor);
        return true;
    }
    return false;
}

} /* namespace katrin */

#endif /* KRAMPEDELECTRICFIELDBUILDER_HH_ */
