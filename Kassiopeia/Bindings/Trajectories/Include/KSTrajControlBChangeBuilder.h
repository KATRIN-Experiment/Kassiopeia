#ifndef Kassiopeia_KSTrajControlBChangeBuilder_h_
#define Kassiopeia_KSTrajControlBChangeBuilder_h_

#include "KSTrajControlBChange.h"
#include "KComplexElement.hh"

using namespace Kassiopeia;
namespace katrin
{

    typedef KComplexElement< KSTrajControlBChange > KSTrajControlBChangeBuilder;

    template< >
    inline bool KSTrajControlBChangeBuilder::AddAttribute( KContainer* aContainer )
    {
        if( aContainer->GetName() == "name" )
        {
            aContainer->CopyTo( fObject, &KNamed::SetName );
            return true;
        }
        if( aContainer->GetName() == "fraction" )
        {
            aContainer->CopyTo( fObject, &KSTrajControlBChange::SetFraction );
            return true;
        }
        return false;
    }

}

#endif
