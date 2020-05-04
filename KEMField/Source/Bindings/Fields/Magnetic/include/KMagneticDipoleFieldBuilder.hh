/*
 * KMagneticDipoleFieldBuilder.hh
 *
 *  Created on: 24 Mar 2016
 *      Author: wolfgang
 */

#ifndef KMAGNETICDIPOLEFIELDBUILDER_HH_
#define KMAGNETICDIPOLEFIELDBUILDER_HH_

#include "KComplexElement.hh"
#include "KEMStreamableThreeVector.hh"
#include "KMagneticDipoleField.hh"

namespace katrin
{

typedef KComplexElement<KEMField::KMagneticDipoleField> KMagneticDipoleFieldBuilder;

template<> inline bool KMagneticDipoleFieldBuilder::AddAttribute(KContainer* aContainer)
{
    if (aContainer->GetName() == "name") {
        std::string name;
        aContainer->CopyTo(name);
        this->SetName(name);
        fObject->SetName(name);
    }
    else if (aContainer->GetName() == "location") {
        KEMField::KEMStreamableThreeVector vec;
        aContainer->CopyTo(vec);
        fObject->SetLocation(vec.GetThreeVector());
    }
    else if (aContainer->GetName() == "moment") {
        KEMField::KEMStreamableThreeVector vec;
        aContainer->CopyTo(vec);
        fObject->SetMoment(vec.GetThreeVector());
    }
    else
        return false;

    return true;
}

} /* namespace katrin */

#endif /* KMAGNETICDIPOLEFIELDBUILDER_HH_ */
