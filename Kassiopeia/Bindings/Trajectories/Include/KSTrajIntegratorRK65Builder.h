#ifndef Kassiopeia_KSTrajIntegratorRK65Builder_h_
#define Kassiopeia_KSTrajIntegratorRK65Builder_h_

#include "KComplexElement.hh"
#include "KSTrajIntegratorRK65.h"

using namespace Kassiopeia;
namespace katrin
{

    typedef KComplexElement< KSTrajIntegratorRK65 > KSTrajIntegratorRK65Builder;

    template< >
    inline bool KSTrajIntegratorRK65Builder::AddAttribute( KContainer* aContainer )
    {
        if( aContainer->GetName() == "name" )
        {
            aContainer->CopyTo( fObject, &KNamed::SetName );
            return true;
        }
        return false;
    }

}

#endif
