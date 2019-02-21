#ifndef Kassiopeia_KSTermMaxStepTimeBuilder_h_
#define Kassiopeia_KSTermMaxStepTimeBuilder_h_

#include "KComplexElement.hh"
#include "KSTermMaxTotalTime.h"

using namespace Kassiopeia;
namespace katrin
{

    typedef KComplexElement< KSTermMaxTotalTime > KSTermMaxTotalTimeBuilder;

    template< >
    inline bool KSTermMaxTotalTimeBuilder::AddAttribute( KContainer* aContainer )
    {
        if( aContainer->GetName() == "name" )
        {
            aContainer->CopyTo( fObject, &KNamed::SetName );
            return true;
        }
        if( aContainer->GetName() == "time" )
        {
            aContainer->CopyTo( fObject, &KSTermMaxTotalTime::SetTime );
            return true;
        }
        return false;
    }

}

#endif
