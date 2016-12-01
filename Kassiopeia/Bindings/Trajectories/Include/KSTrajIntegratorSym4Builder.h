#ifndef Kassiopeia_KSTrajIntegratorSym4Builder_h_
#define Kassiopeia_KSTrajIntegratorSym4Builder_h_

#include "KComplexElement.hh"
#include "KSTrajIntegratorSym4.h"

using namespace Kassiopeia;
namespace katrin
{

    typedef KComplexElement< KSTrajIntegratorSym4 > KSTrajIntegratorSym4Builder;

    template< >
    inline bool KSTrajIntegratorSym4Builder::AddAttribute( KContainer* aContainer )
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
