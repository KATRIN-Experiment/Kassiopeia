/*
 * KMagneticDipoleFieldBuilder.hh
 *
 *  Created on: 24 Mar 2016
 *      Author: wolfgang
 */

#ifndef KEMFIELD_SOURCE_2_0_PLUGINS_BINDINGS_FIELDS_MAGNETIC_INCLUDE_KMAGNETICDIPOLEFIELDBUILDER_HH_
#define KEMFIELD_SOURCE_2_0_PLUGINS_BINDINGS_FIELDS_MAGNETIC_INCLUDE_KMAGNETICDIPOLEFIELDBUILDER_HH_

#include "KComplexElement.hh"
#include "KMagneticDipoleField.hh"
#include "KEMStreamableThreeVector.hh"

namespace katrin {

typedef KComplexElement< KEMField::KMagneticDipoleField > KMagneticDipoleFieldBuilder;

template< >
inline bool KMagneticDipoleFieldBuilder::AddAttribute( KContainer* aContainer )
{
    if( aContainer->GetName() == "name" )
    {
        std::string name;
        aContainer->CopyTo(name);
        this->SetName(name);
        fObject->SetName(name);
    }
    else if( aContainer->GetName() == "location" )
    {
        KEMField::KEMStreamableThreeVector vec;
        aContainer->CopyTo(vec);
        fObject->SetLocation(vec.GetThreeVector());
    }
    else if( aContainer->GetName() == "moment" )
    {
        KEMField::KEMStreamableThreeVector vec;
        aContainer->CopyTo(vec);
        fObject->SetMoment(vec.GetThreeVector());
    }
    else return false;

    return true;
}

} /* namespace katrin */

#endif /* KEMFIELD_SOURCE_2_0_PLUGINS_BINDINGS_FIELDS_MAGNETIC_INCLUDE_KMAGNETICDIPOLEFIELDBUILDER_HH_ */
