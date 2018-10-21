#ifndef Kassiopeia_KSIntSurfaceSpinFlipBuilder_h_
#define Kassiopeia_KSIntSurfaceSpinFlipBuilder_h_

#include "KComplexElement.hh"
#include "KSIntSurfaceSpinFlip.h"

using namespace Kassiopeia;
namespace katrin
{

    typedef KComplexElement< KSIntSurfaceSpinFlip > KSIntSurfaceSpinFlipBuilder;

    template< >
    inline bool KSIntSurfaceSpinFlipBuilder::AddAttribute( KContainer* aContainer )
    {
        if( aContainer->GetName() == "name" )
        {
            aContainer->CopyTo( fObject, &KNamed::SetName );
            return true;
        }
        if( aContainer->GetName() == "probability" )
        {
            aContainer->CopyTo( fObject, &KSIntSurfaceSpinFlip::SetProbability );
            return true;
        }
        return false;
    }

}

#endif
