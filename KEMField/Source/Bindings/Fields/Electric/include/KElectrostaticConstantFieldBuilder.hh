#ifndef KELECTROSTATICCONSTANTFIELDBUILDER_HH_
#define KELECTROSTATICCONSTANTFIELDBUILDER_HH_

#include "KComplexElement.hh"
#include "KEMStreamableThreeVector.hh"
#include "KElectrostaticConstantField.hh"

#include <iostream>

namespace katrin
{

typedef KComplexElement<KEMField::KElectrostaticConstantField> KElectrostaticConstantFieldBuilder;

template<> inline bool KElectrostaticConstantFieldBuilder::AddAttribute(KContainer* aContainer)
{
    if (aContainer->GetName() == "name") {
        std::string name;
        aContainer->CopyTo(name);
        this->SetName(name);
        fObject->SetName(name);
        return true;
    }
    else if (aContainer->GetName() == "field") {
        KEMField::KEMStreamableThreeVector vec;
        aContainer->CopyTo(vec);
        fObject->SetField(vec.GetThreeVector());
        return true;
    }
    else if (aContainer->GetName() == "location") {
        KEMField::KEMStreamableThreeVector vec;
        aContainer->CopyTo(vec);
        fObject->SetLocation(vec.GetThreeVector());
        return true;
    }
    else if (aContainer->GetName() == "offset_potential") {
        aContainer->CopyTo(fObject, &KEMField::KElectrostaticConstantField::SetPotentialOffset);
        return true;
    }
    return false;
}

template<> inline bool KElectrostaticConstantFieldBuilder::AddElement(KContainer* /*aContainer*/)
{
    return false;
}

}  // namespace katrin


#endif /* KELECTROSTATICCONSTANTFIELDBUILDER_HH_ */
