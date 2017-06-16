/*
 * KMagnetostaticConstantFieldBuilder.hh
 *
 *  Created on: 23 Mar 2016
 *      Author: wolfgang
 */

#ifndef KEMFIELD_SOURCE_2_0_PLUGINS_BINDINGS_FIELDS_MAGNETIC_INCLUDE_KMAGNETOSTATICCONSTANTFIELDBUILDER_HH_
#define KEMFIELD_SOURCE_2_0_PLUGINS_BINDINGS_FIELDS_MAGNETIC_INCLUDE_KMAGNETOSTATICCONSTANTFIELDBUILDER_HH_

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

#endif /* KEMFIELD_SOURCE_2_0_PLUGINS_BINDINGS_FIELDS_MAGNETIC_INCLUDE_KMAGNETOSTATICCONSTANTFIELDBUILDER_HH_ */
