#ifndef Kassiopeia_KSTrajTermDriftBuilder_h_
#define Kassiopeia_KSTrajTermDriftBuilder_h_

#include "KComplexElement.hh"
#include "KSTrajTermDrift.h"

using namespace Kassiopeia;
namespace katrin
{

    typedef KComplexElement< KSTrajTermDrift > KSTrajTermDriftBuilder;

    template< >
    inline bool KSTrajTermDriftBuilder::AddAttribute( KContainer* aContainer )
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
