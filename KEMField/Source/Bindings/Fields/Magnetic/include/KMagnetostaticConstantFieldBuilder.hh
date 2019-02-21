/*
 * KMagnetostaticConstantFieldBuilder.hh
 *
 *  Created on: 23 Mar 2016
 *      Author: wolfgang
 */

#ifndef KMAGNETOSTATICCONSTANTFIELDBUILDER_HH_
#define KMAGNETOSTATICCONSTANTFIELDBUILDER_HH_

#include "KComplexElement.hh"
#include "KMagnetostaticConstantField.hh"
#include "KEMStreamableThreeVector.hh"

namespace katrin {

typedef KComplexElement< KEMField::KMagnetostaticConstantField > KSFieldMagneticConstantBuilder;

template< >
inline bool KSFieldMagneticConstantBuilder::AddAttribute( KContainer* aContainer )
{
    if( aContainer->GetName() == "name" )
    {
        std::string name;
        aContainer->CopyTo(name);
        this->SetName(name);
        fObject->SetName(name);
    }
    else if( aContainer->GetName() == "field" )
    {
        KEMField::KEMStreamableThreeVector vec;
        aContainer->CopyTo(vec);
        fObject->SetField(vec.GetThreeVector());
    }
    else
    {
        return false;
    }
    return true;
}

} /* namespace katrin */

#endif /* KMAGNETOSTATICCONSTANTFIELDBUILDER_HH_ */
