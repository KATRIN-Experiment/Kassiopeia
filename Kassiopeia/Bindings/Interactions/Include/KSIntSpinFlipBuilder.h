#ifndef Kassiopeia_KSIntSpinFlipBuilder_h_
#define Kassiopeia_KSIntSpinFlipBuilder_h_

#include "KComplexElement.hh"
#include "KSIntSpinFlip.h"

using namespace Kassiopeia;
namespace katrin
{

    typedef KComplexElement< KSIntSpinFlip > KSIntSpinFlipBuilder;

    template< >
    inline bool KSIntSpinFlipBuilder::AddAttribute( KContainer* aContainer )
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
