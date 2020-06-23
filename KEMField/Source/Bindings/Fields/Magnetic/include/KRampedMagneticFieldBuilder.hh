/*
 * KRampedMagneticFieldBuilder.hh
 *
 *  Created on: 8 Apr 2016
 *      Author: wolfgang
 */

#ifndef KRAMPEDMAGNETICFIELDBUILDER_HH_
#define KRAMPEDMAGNETICFIELDBUILDER_HH_

#include "KComplexElement.hh"
#include "KEMBindingsMessage.hh"
#include "KRampedMagneticField.hh"
#include "KToolbox.h"

namespace katrin
{

typedef KComplexElement<KEMField::KRampedMagneticField> KRampedMagneticFieldBuilder;

template<> inline bool KRampedMagneticFieldBuilder::AddAttribute(KContainer* aContainer)
{
    if (aContainer->GetName() == "name") {
        std::string name;
        aContainer->CopyTo(name);
        this->SetName(name);
        fObject->SetName(name);
    }
    else if (aContainer->GetName() == "root_field") {
        std::string fieldName = aContainer->AsReference<std::string>();
        auto* field = katrin::KToolbox::GetInstance().Get<KEMField::KMagneticField>(fieldName);
        fObject->SetMagneticField(field);
    }
    else if (aContainer->GetName() == "ramping_type") {
        std::string tFlag = aContainer->AsReference<std::string>();
        if (tFlag == std::string("linear"))
            fObject->SetRampingType(KEMField::KRampedMagneticField::rtLinear);
        else if (tFlag == std::string("exponential"))
            fObject->SetRampingType(KEMField::KRampedMagneticField::rtExponential);
        else if (tFlag == std::string("inversion"))
            fObject->SetRampingType(KEMField::KRampedMagneticField::rtInversion);
        else if (tFlag == std::string("inversion2"))
            fObject->SetRampingType(KEMField::KRampedMagneticField::rtInversion2);
        else if (tFlag == std::string("flipbox"))
            fObject->SetRampingType(KEMField::KRampedMagneticField::rtFlipBox);
        else {
            BINDINGMSG(eError) << "ramped_magnetic_field got unknown ramping type: " << tFlag << "." << eom;
        }
    }
    else if (aContainer->GetName() == "num_cycles") {
        aContainer->CopyTo(fObject, &KEMField::KRampedMagneticField::SetNumCycles);
    }
    else if (aContainer->GetName() == "ramp_up_delay") {
        aContainer->CopyTo(fObject, &KEMField::KRampedMagneticField::SetRampUpDelay);
    }
    else if (aContainer->GetName() == "ramp_down_delay") {
        aContainer->CopyTo(fObject, &KEMField::KRampedMagneticField::SetRampDownDelay);
    }
    else if (aContainer->GetName() == "ramp_up_time") {
        aContainer->CopyTo(fObject, &KEMField::KRampedMagneticField::SetRampUpTime);
    }
    else if (aContainer->GetName() == "ramp_down_time") {
        aContainer->CopyTo(fObject, &KEMField::KRampedMagneticField::SetRampDownTime);
    }
    else if (aContainer->GetName() == "time_constant") {
        aContainer->CopyTo(fObject, &KEMField::KRampedMagneticField::SetTimeConstant);
    }
    else if (aContainer->GetName() == "time_constant_2") {
        aContainer->CopyTo(fObject, &KEMField::KRampedMagneticField::SetTimeConstant2);
    }
    else if (aContainer->GetName() == "time_scaling") {
        aContainer->CopyTo(fObject, &KEMField::KRampedMagneticField::SetTimeScalingFactor);
    }
    else
        return false;
    return true;
}

} /* namespace katrin */

#endif /* KRAMPEDMAGNETICFIELDBUILDER_HH_ */
