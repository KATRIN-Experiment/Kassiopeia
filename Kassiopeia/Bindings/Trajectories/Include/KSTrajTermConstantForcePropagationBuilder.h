#ifndef KSTRAJTERMCONSTANTFORCEPROPAGATIONBUILDER_H
#define KSTRAJTERMCONSTANTFORCEPROPAGATIONBUILDER_H

#include "KComplexElement.hh"
#include "KSTrajTermConstantForcePropagation.h"


using namespace Kassiopeia;
namespace katrin
{

    typedef KComplexElement< KSTrajTermConstantForcePropagation > KSTrajTermConstantForcePropagationBuilder;

    template< >
    inline bool KSTrajTermConstantForcePropagationBuilder::AddAttribute( KContainer* aContainer )
    {
        if( aContainer->GetName() == "name" )
        {
            aContainer->CopyTo( fObject, &KNamed::SetName );
            return true;
        }
        if( (aContainer->GetName() == "force")  )
        {
            KThreeVector* tForce = NULL;
            aContainer->ReleaseTo( tForce );
            fObject->SetForce( *tForce );
            delete tForce;
            return true;
        }
        return false;
    }

}

#endif // KSTRAJTERMCONSTANTFORCEPROPAGATIONBUILDER_H
