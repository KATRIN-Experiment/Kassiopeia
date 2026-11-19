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

//! takes an magnetic field and modulates it with the time according
//! to the selected profile.
//! This field does not consider electrodynamics and just scales the
//! underlying static field. It is only for sufficiently slowly varying
//! fields. Don't simulate an antenna with it!
//!
//! For details please see the implementation of the different profiles in
//! the code (the file is KRampedMagneticField.cc).
//! @example
//! <ramped_magnetic_field
//!     name="magnetic_ramped"
//!     root_field="magnetic_dipole"
//!     ramping_type="linear"
//!     num_cycles="10"
//!     ramp_up_delay="-0.5"
//!     ramp_down_delay="0.5"
//!     ramp_up_time="1.0"
//!     ramp_down_time="1.0"
//!     time_constant="2.0"
//!     time_scaling="1.0"
//! />
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
        std::string fieldName = aContainer->AsString();
        auto* field = katrin::KToolbox::GetInstance().Get<KEMField::KMagneticField>(fieldName);
        fObject->SetMagneticField(field);
    }
    else if (aContainer->GetName() == "ramping_type") {
        std::string tFlag = aContainer->AsString();
        if (tFlag == "linear")
            fObject->SetRampingType(KEMField::KRampedMagneticField::rtLinear);
        else if (tFlag == "exponential")
            fObject->SetRampingType(KEMField::KRampedMagneticField::rtExponential);
        else if (tFlag == "inversion")
            fObject->SetRampingType(KEMField::KRampedMagneticField::rtInversion);
        else if (tFlag == "inversion2")
            fObject->SetRampingType(KEMField::KRampedMagneticField::rtInversion2);
        else if (tFlag == "flipbox")
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
