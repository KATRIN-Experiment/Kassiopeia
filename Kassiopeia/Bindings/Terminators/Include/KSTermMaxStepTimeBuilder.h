#ifndef Kassiopeia_KSTermMaxStepTimeBuilder_h_
#define Kassiopeia_KSTermMaxStepTimeBuilder_h_

#include "KComplexElement.hh"
#include "KSTermMaxStepTime.h"

using namespace Kassiopeia;
namespace katrin
{

    typedef KComplexElement< KSTermMaxStepTime > KSTermMaxStepTimeBuilder;

    template< >
    inline bool KSTermMaxStepTimeBuilder::AddAttribute( KContainer* aContainer )
    {
        if( aContainer->GetName() == "name" )
        {
            aContainer->CopyTo( fObject, &KNamed::SetName );
            return true;
        }
        if( aContainer->GetName() == "time" )
        {
            aContainer->CopyTo( fObject, &KSTermMaxStepTime::SetTime );
            return true;
        }
        return false;
    }

}

#endif
