#ifndef Kassiopeia_KSTermStepsizeBuilder_h_
#define Kassiopeia_KSTermStepsizeBuilder_h_

#include "KComplexElement.hh"
#include "KSTermStepsize.h"

using namespace Kassiopeia;
namespace katrin
{

    typedef KComplexElement< KSTermStepsize > KSTermStepsizeBuilder;

    template< >
    inline bool KSTermStepsizeBuilder::AddAttribute( KContainer* aContainer )
    {
        if( aContainer->GetName() == "name" )
        {
            aContainer->CopyTo( fObject, &KNamed::SetName );
            return true;
        }
        if( aContainer->GetName() == "min_length" )
        {
            aContainer->CopyTo( fObject, &KSTermStepsize::SetLowerLimit );
            return true;
        }
        if( aContainer->GetName() == "max_length" )
        {
            aContainer->CopyTo( fObject, &KSTermStepsize::SetUpperLimit );
            return true;
        }
        return false;
    }

}

#endif
