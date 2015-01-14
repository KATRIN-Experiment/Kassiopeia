#ifndef Kassiopeia_KSTrajIntegratorRK54Builder_h_
#define Kassiopeia_KSTrajIntegratorRK54Builder_h_

#include "KComplexElement.hh"
#include "KSTrajIntegratorRK54.h"

using namespace Kassiopeia;
namespace katrin
{

    typedef KComplexElement< KSTrajIntegratorRK54 > KSTrajIntegratorRK54Builder;

    template< >
    inline bool KSTrajIntegratorRK54Builder::AddAttribute( KContainer* aContainer )
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
