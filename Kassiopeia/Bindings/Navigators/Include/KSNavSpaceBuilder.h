#ifndef Kassiopeia_KSNavSpaceBuilder_h_
#define Kassiopeia_KSNavSpaceBuilder_h_

#include "KComplexElement.hh"
#include "KSNavSpace.h"

using namespace Kassiopeia;
namespace katrin
{

    typedef KComplexElement< KSNavSpace > KSNavSpaceBuilder;

    template< >
    inline bool KSNavSpaceBuilder::AddAttribute( KContainer* aContainer )
    {
        if( aContainer->GetName() == "name" )
        {
            aContainer->CopyTo( fObject, &KNamed::SetName );
            return true;
        }
        if( aContainer->GetName() == "enter_split" )
        {
            aContainer->CopyTo( fObject, &KSNavSpace::SetEnterSplit );
            return true;
        }
        if( aContainer->GetName() == "exit_split" )
        {
            aContainer->CopyTo( fObject, &KSNavSpace::SetExitSplit );
            return true;
        }
        if( aContainer->GetName() == "tolerance" )
        {
            aContainer->CopyTo( fObject, &KSNavSpace::SetTolerance );
            return true;
        }
        return false;
    }

}
#endif
