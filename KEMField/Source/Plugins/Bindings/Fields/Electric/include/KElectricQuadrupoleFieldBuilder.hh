/*
 * KElectricQuadrupoleFieldBuilder.hh
 *
 *  Created on: 30 Jul 2015
 *      Author: wolfgang
 */

#ifndef KEMFIELD_SOURCE_2_0_PLUGINS_BINDINGS_FIELDS_ELECTRIC_INCLUDE_KELECTRICQUADRUPOLEFIELDBUILDER_HH_
#define KEMFIELD_SOURCE_2_0_PLUGINS_BINDINGS_FIELDS_ELECTRIC_INCLUDE_KELECTRICQUADRUPOLEFIELDBUILDER_HH_

#include "KComplexElement.hh"

#include "KElectricQuadrupoleField.hh"

#include "KEMStreamableThreeVector.hh"

namespace katrin {

typedef KComplexElement< KEMField::KElectricQuadrupoleField >
KElectricQuadrupoleFieldBuilder;

template< >
inline bool KElectricQuadrupoleFieldBuilder::AddAttribute( KContainer* aContainer )
{
    if( aContainer->GetName() == "name" )
    {
		std::string name;
		aContainer->CopyTo(name);
        fObject->SetName(name);
		this->SetName(name);
        return true;
    }
    if( aContainer->GetName() == "location" )
    {
    	KEMField::KEMStreamableThreeVector vector;
        aContainer->CopyTo( vector );
        fObject->SetLocation(KEMField::KPosition(vector.GetThreeVector()));
        return true;
    }
    if( aContainer->GetName() == "strength" )
    {
        aContainer->CopyTo( fObject, &KEMField::KElectricQuadrupoleField::SetStrength );
        return true;
    }
    if( aContainer->GetName() == "length" )
    {
        aContainer->CopyTo( fObject, &KEMField::KElectricQuadrupoleField::SetLength );
        return true;
    }
    if( aContainer->GetName() == "radius" )
    {
        aContainer->CopyTo( fObject, &KEMField::KElectricQuadrupoleField::SetRadius );
        return true;
    }
    return false;
}

} /* namespace katrin */

#endif /* KEMFIELD_SOURCE_2_0_PLUGINS_BINDINGS_FIELDS_ELECTRIC_INCLUDE_KELECTRICQUADRUPOLEFIELDBUILDER_HH_ */
