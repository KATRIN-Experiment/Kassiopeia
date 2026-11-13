/*
 * KElectricQuadrupoleFieldBuilder.hh
 *
 *  Created on: 30 Jul 2015
 *      Author: wolfgang
 */

#ifndef KELECTRICQUADRUPOLEFIELDBUILDER_HH_
#define KELECTRICQUADRUPOLEFIELDBUILDER_HH_

#include "KComplexElement.hh"
#include "KEMStreamableThreeVector.hh"
#include "KElectricQuadrupoleField.hh"

namespace katrin
{

//! provides a perfect quadrupole electric field.
//! equipotential surfaces are hyperboloids of revolution about the z axis, with asymptotic cones at zero potential.
//! the potential in cylindrical coordinates obeys:
//!
//! V(z,r,phi) = V0 * (z^2 - r^2 / 2) / (z0^2 + r0^2 / 2)
//! @example <electric_quadrupole_field name="electric_quadrupole" location="0. 0. 0." strength="-10000." length="1.e-2" radius="1.e-2"/>
typedef KComplexElement<KEMField::KElectricQuadrupoleField> KElectricQuadrupoleFieldBuilder;

template<> inline bool KElectricQuadrupoleFieldBuilder::AddAttribute(KContainer* aContainer)
{
    if (aContainer->GetName() == "name") {
        std::string name;
        aContainer->CopyTo(name);
        fObject->SetName(name);
        this->SetName(name);
        return true;
    }
    if (aContainer->GetName() == "location") {
        KEMField::KEMStreamableThreeVector vector;
        aContainer->CopyTo(vector);
        fObject->SetLocation(KEMField::KPosition(vector.GetThreeVector()));
        return true;
    }
    if (aContainer->GetName() == "strength") {
        aContainer->CopyTo(fObject, &KEMField::KElectricQuadrupoleField::SetStrength);
        return true;
    }
    if (aContainer->GetName() == "length") {
        aContainer->CopyTo(fObject, &KEMField::KElectricQuadrupoleField::SetLength);
        return true;
    }
    if (aContainer->GetName() == "radius") {
        aContainer->CopyTo(fObject, &KEMField::KElectricQuadrupoleField::SetRadius);
        return true;
    }
    return false;
}

} /* namespace katrin */

#endif /* KELECTRICQUADRUPOLEFIELDBUILDER_HH_ */
