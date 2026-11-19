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

//! takes an electric field and modulates it with the time according
//! to the selected profile.
//! This field does not consider electrodynamics and just scales the
//! underlying static field. It is only for sufficiently slowly varying
//! fields. Don't simulate an antenna with it!
//!
//! For details please see the implementation of the different profiles in
//! the code (the file is KRampedElectricField.cc).
//! @example
//! <ramped_electric_field
//!     name="ramped_electric_field"
//!     root_field="electric_quadrupole"
//!     ramping_type="linear"
//!     num_cycles="10"
//!     ramp_up_delay="-0.1"
//!     ramp_down_delay="0.1"
//!     ramp_up_time="0.2"
//!     ramp_down_time="0.2"
//!     time_constant="2.0"
//!     time_scaling="1.0"
//!
//! />
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
        std::string fieldName = aContainer->AsString();
        auto* field = katrin::KToolbox::GetInstance().Get<KEMField::KElectricField>(fieldName);
        fObject->SetRootElectricField(field);
        return true;
    }
    if (aContainer->GetName() == "ramping_type") {
        std::string tFlag = aContainer->AsString();
        if (tFlag == "linear")
            fObject->SetRampingType(KEMField::KRampedElectricField::rtLinear);
        else if (tFlag == "exponential")
            fObject->SetRampingType(KEMField::KRampedElectricField::rtExponential);
        else if (tFlag == "sinus")
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
