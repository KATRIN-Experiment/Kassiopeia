#ifndef Kassiopeia_KSTermMaxStepsBuilder_h_
#define Kassiopeia_KSTermMaxStepsBuilder_h_

#include "KComplexElement.hh"
#include "KSTermMaxSteps.h"

using namespace Kassiopeia;
namespace katrin
{

    typedef KComplexElement< KSTermMaxSteps > KSTermMaxStepsBuilder;

    template< >
    inline bool KSTermMaxStepsBuilder::AddAttribute( KContainer* aContainer )
    {
        if( aContainer->GetName() == "name" )
        {
            aContainer->CopyTo( fObject, &KNamed::SetName );
            return true;
        }
        if( aContainer->GetName() == "steps" )
        {
            aContainer->CopyTo( fObject, &KSTermMaxSteps::SetMaxSteps );
            return true;
        }
        return false;
    }

}
#endif
