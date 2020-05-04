/*
 * KMagnetostaticConstantFieldBuilder.hh
 *
 *  Created on: 23 Mar 2016
 *      Author: wolfgang
 */

#ifndef KMAGNETOSTATICCONSTANTFIELDBUILDER_HH_
#define KMAGNETOSTATICCONSTANTFIELDBUILDER_HH_

#include "KComplexElement.hh"
#include "KEMStreamableThreeVector.hh"
#include "KMagnetostaticConstantField.hh"

namespace katrin
{

typedef KComplexElement<KEMField::KMagnetostaticConstantField> KSFieldMagneticConstantBuilder;

template<> inline bool KSFieldMagneticConstantBuilder::AddAttribute(KContainer* aContainer)
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
    return false;
}

} /* namespace katrin */

#endif /* KMAGNETOSTATICCONSTANTFIELDBUILDER_HH_ */
