#ifndef KELECTROSTATICLINEARFIELDBUILDER_HH_
#define KELECTROSTATICLINEARFIELDBUILDER_HH_

#include "KComplexElement.hh"
#include "KEMStreamableThreeVector.hh"
#include "KElectrostaticLinearField.hh"

#include <iostream>

namespace katrin
{

//! provides a perfect linear electric field.
//! the potential in cylindrical coordinates obeys:
//!
//! V(z,r,phi) = V0 + (V1 - V0) / (z1 - z0) * (z - z0)
//! @example <linear_electric_field name="electric_linear" z1="0" z2="1.e-2" U1="100" U2="0"/>
typedef KComplexElement<KEMField::KElectrostaticLinearField> KElectrostaticLinearFieldBuilder;

template<> inline bool KElectrostaticLinearFieldBuilder::AddAttribute(KContainer* aContainer)
{
    if (aContainer->GetName() == "name") {
        std::string name;
        aContainer->CopyTo(name);
        this->SetName(name);
        fObject->SetName(name);
        return true;
    }
    if (aContainer->GetName() == "U1") {
        aContainer->CopyTo(fObject, &KEMField::KElectrostaticLinearField::SetPotential1);
        return true;
    }
    if (aContainer->GetName() == "U2") {
        aContainer->CopyTo(fObject, &KEMField::KElectrostaticLinearField::SetPotential2);
        return true;
    }
    if (aContainer->GetName() == "z1") {
        aContainer->CopyTo(fObject, &KEMField::KElectrostaticLinearField::SetZ1);
        return true;
    }
    if (aContainer->GetName() == "z2") {
        aContainer->CopyTo(fObject, &KEMField::KElectrostaticLinearField::SetZ2);
        return true;
    }
    if (aContainer->GetName() == "surface") {
        // TODO
        return false;
    }
    return false;
}

template<> inline bool KElectrostaticLinearFieldBuilder::AddElement(KContainer* /*aContainer*/)
{
    return false;
}

}  // namespace katrin


#endif /* KELECTROSTATICCONSTANTFIELDBUILDER_HH_ */
