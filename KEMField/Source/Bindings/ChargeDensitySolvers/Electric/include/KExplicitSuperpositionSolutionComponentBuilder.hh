/*
 * KExplicitSuperpositionSolutionComponentBuilder.hh
 *
 *  Created on: 27 Jun 2016
 *      Author: wolfgang
 */

#ifndef KEXPLICITSUPERPOSITIONSOLUTIONCOMPONENTBUILDER_HH_
#define KEXPLICITSUPERPOSITIONSOLUTIONCOMPONENTBUILDER_HH_

#include "KComplexElement.hh"
#include "KExplicitSuperpositionSolutionComponent.hh"

namespace katrin {

typedef KComplexElement< KEMField::KExplicitSuperpositionSolutionComponent > KExplicitSuperpositionSolutionComponentBuilder;

template< >
inline bool KExplicitSuperpositionSolutionComponentBuilder::AddAttribute( KContainer* aContainer )
{
    if( aContainer->GetName() == "name" )
    {
        aContainer->CopyTo( fObject->name );
        return true;
    }
    if( aContainer->GetName() == "scale" )
    {
        aContainer->CopyTo( fObject->scale );
        return true;
    }
    if( aContainer->GetName() == "hash" )
    {
        aContainer->CopyTo( fObject->hash );
        return true;
    }
    return false;

}

} /* namespace katrin */

#endif /* KEXPLICITSUPERPOSITIONSOLUTIONCOMPONENTBUILDER_HH_ */
