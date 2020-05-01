/*
 * KRampedElectric2FieldBuilder.hh
 *
 *  Created on: 16 Jun 2016
 *      Author: wolfgang
 */

#ifndef KRAMPEDELECTRIC2FIELDBUILDER_HH_
#define KRAMPEDELECTRIC2FIELDBUILDER_HH_

#include "KComplexElement.hh"
#include "KRampedElectric2Field.hh"
#include "KToolbox.h"

namespace katrin
{

typedef KComplexElement<KEMField::KRampedElectric2Field> KRampedElectric2FieldBuilder;

template<> inline bool KRampedElectric2FieldBuilder::AddAttribute(KContainer* aContainer)
{
    if (aContainer->GetName() == "name") {
        std::string name;
        aContainer->CopyTo(name);
        this->SetName(name);
        fObject->SetName(name);
        return true;
    }
    if (aContainer->GetName() == "root_field_1") {
        std::string fieldName = aContainer->AsReference<std::string>();
        auto* field = katrin::KToolbox::GetInstance().Get<KEMField::KElectricField>(fieldName);
        fObject->SetRootElectricField1(field);
        return true;
    }
    if (aContainer->GetName() == "root_field_2") {
        std::string fieldName = aContainer->AsReference<std::string>();
        auto* field = katrin::KToolbox::GetInstance().Get<KEMField::KElectricField>(fieldName);
        fObject->SetRootElectricField2(field);
        return true;
    }
    if (aContainer->GetName() == "ramping_type") {
        std::string tFlag = aContainer->AsReference<std::string>();
        if (tFlag == std::string("linear"))
            fObject->SetRampingType(KEMField::KRampedElectric2Field::rtLinear);
        else if (tFlag == std::string("exponential"))
            fObject->SetRampingType(KEMField::KRampedElectric2Field::rtExponential);
        else if (tFlag == std::string("sinus"))
            fObject->SetRampingType(KEMField::KRampedElectric2Field::rtSinus);
        else if (tFlag == std::string("square"))
            fObject->SetRampingType(KEMField::KRampedElectric2Field::rtSquare);
        else if (tFlag == std::string("focus"))
            fObject->SetRampingType(KEMField::KRampedElectric2Field::rtFocus);
        else if (tFlag == std::string("focus_experimental"))
            fObject->SetRampingType(KEMField::KRampedElectric2Field::rtFocusExperimental);
        return true;
    }
    if (aContainer->GetName() == "num_cycles") {
        aContainer->CopyTo(fObject, &KEMField::KRampedElectric2Field::SetNumCycles);
        return true;
    }
    if (aContainer->GetName() == "ramp_up_delay") {
        aContainer->CopyTo(fObject, &KEMField::KRampedElectric2Field::SetRampUpDelay);
        return true;
    }
    if (aContainer->GetName() == "ramp_down_delay") {
        aContainer->CopyTo(fObject, &KEMField::KRampedElectric2Field::SetRampDownDelay);
        return true;
    }
    if (aContainer->GetName() == "ramp_up_time") {
        aContainer->CopyTo(fObject, &KEMField::KRampedElectric2Field::SetRampUpTime);
        return true;
    }
    if (aContainer->GetName() == "ramp_down_time") {
        aContainer->CopyTo(fObject, &KEMField::KRampedElectric2Field::SetRampDownTime);
        return true;
    }
    if (aContainer->GetName() == "time_constant") {
        aContainer->CopyTo(fObject, &KEMField::KRampedElectric2Field::SetTimeConstant);
        return true;
    }
    if (aContainer->GetName() == "time_scaling") {
        aContainer->CopyTo(fObject, &KEMField::KRampedElectric2Field::SetTimeScalingFactor);
        return true;
    }
    if (aContainer->GetName() == "focus_time") {
        aContainer->CopyTo(fObject, &KEMField::KRampedElectric2Field::SetFocusTime);
        return true;
    }
    if (aContainer->GetName() == "focus_exponent") {
        aContainer->CopyTo(fObject, &KEMField::KRampedElectric2Field::SetFocusExponent);
        return true;
    }
    if (aContainer->GetName() == "small_spectrometer") {
        aContainer->CopyTo(fObject, &KEMField::KRampedElectric2Field::SetSmall);
        return true;
    }
    return false;
}

} /* namespace katrin */

#endif /* KRAMPEDELECTRIC2FIELDBUILDER_HH_ */
