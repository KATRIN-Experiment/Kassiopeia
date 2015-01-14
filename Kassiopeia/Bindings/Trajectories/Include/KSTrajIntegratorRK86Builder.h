#ifndef Kassiopeia_KSTrajIntegratorRK86Builder_h_
#define Kassiopeia_KSTrajIntegratorRK86Builder_h_

#include "KComplexElement.hh"
#include "KSTrajIntegratorRK86.h"

using namespace Kassiopeia;
namespace katrin
{

    typedef KComplexElement< KSTrajIntegratorRK86 > KSTrajIntegratorRK86Builder;

    template< >
    inline bool KSTrajIntegratorRK86Builder::AddAttribute( KContainer* aContainer )
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
