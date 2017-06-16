#ifndef Kassiopeia_KSGenValueZFrustrumBuilder_h_
#define Kassiopeia_KSGenValueZFrustrumBuilder_h_

#include "KComplexElement.hh"
#include "KSGenValueZFrustrum.h"

using namespace Kassiopeia;
namespace katrin
{

    typedef KComplexElement< KSGenValueZFrustrum > KSGenValueZFrustrumBuilder;

    template< >
    inline bool KSGenValueZFrustrumBuilder::AddAttribute( KContainer* aContainer )
    {
        if( aContainer->GetName() == "name" )
        {
            aContainer->CopyTo( fObject, &KNamed::SetName );
            return true;
        }
        if( aContainer->GetName() == "r1" )
        {
            aContainer->CopyTo( fObject, &KSGenValueZFrustrum::Setr1 );
            return true;
        }
        if( aContainer->GetName() == "r2" )
        {
            aContainer->CopyTo( fObject, &KSGenValueZFrustrum::Setr2 );
            return true;
        }
        if( aContainer->GetName() == "z1" )
        {
            aContainer->CopyTo( fObject, &KSGenValueZFrustrum::Setz1 );
            return true;
        }
        if( aContainer->GetName() == "z2" )
        {
            aContainer->CopyTo( fObject, &KSGenValueZFrustrum::Setz2 );
            return true;
        }
        return false;
    }

}

#endif
