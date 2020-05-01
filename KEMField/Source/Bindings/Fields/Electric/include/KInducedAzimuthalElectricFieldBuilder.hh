/*
 * KInducedAzimuthalElectricFieldBuilder.hh
 *
 *  Created on: 15 Apr 2016
 *      Author: wolfgang
 */

#ifndef KINDUCEDAZIMUTHALELECTRICFIELDBUILDER_HH_
#define KINDUCEDAZIMUTHALELECTRICFIELDBUILDER_HH_

#include "KComplexElement.hh"
#include "KEMBindingsMessage.hh"
#include "KInducedAzimuthalElectricField.hh"
#include "KRampedMagneticField.hh"
#include "KToolbox.h"

namespace katrin
{

typedef KComplexElement<KEMField::KInducedAzimuthalElectricField> KInducedAzimuthalElectricFieldBuilder;

template<> inline bool KInducedAzimuthalElectricFieldBuilder::AddAttribute(KContainer* aContainer)
{
    if (aContainer->GetName() == "name") {
        std::string name;
        aContainer->CopyTo(name);
        this->SetName(name);
        fObject->SetName(name);
    }
    else if (aContainer->GetName() == "root_field") {
        std::string fieldName = aContainer->AsReference<std::string>();
        // toolbox contains only entries of type KMagneticField, so we have to
        // do an ugly downcast
        auto* magneticField = katrin::KToolbox::GetInstance().Get<KEMField::KMagneticField>(fieldName);
        auto rampedMagneticField = dynamic_cast<KEMField::KRampedMagneticField*>(magneticField);
        if (rampedMagneticField)
            fObject->SetMagneticField(rampedMagneticField);
        else {
            BINDINGMSG(eError) << "induced_azimuthal_electric_field only"
                                  " accepts ramped magnetic fields. <"
                               << fieldName
                               << "> is a"
                                  " different kind of magnetic field."
                               << eom;
        }
    }
    else
        return false;
    return true;
}

} /* namespace katrin */

#endif /* KINDUCEDAZIMUTHALELECTRICFIELDBUILDER_HH_ */
