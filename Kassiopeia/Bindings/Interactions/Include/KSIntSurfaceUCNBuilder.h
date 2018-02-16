#ifndef Kassiopeia_KSIntSurfaceUCNBuilder_h_
#define Kassiopeia_KSIntSurfaceUCNBuilder_h_

#include "KComplexElement.hh"
#include "KSIntSurfaceUCN.h"

using namespace Kassiopeia;
namespace katrin
{

    typedef KComplexElement< KSIntSurfaceUCN > KSIntSurfaceUCNBuilder;

    template< >
    inline bool KSIntSurfaceUCNBuilder::AddAttribute( KContainer* aContainer )
    {
        if( aContainer->GetName() == "name" )
        {
            aContainer->CopyTo( fObject, &KNamed::SetName );
            return true;
        }
        if( aContainer->GetName() == "transmission_probability" )
        {
            aContainer->CopyTo( fObject, &KSIntSurfaceUCN::SetProbability );
            return true;
        }
        if( aContainer->GetName() == "spin_flip_probability" )
        {
            aContainer->CopyTo( fObject, &KSIntSurfaceUCN::SetSpinFlipProbability );
            return true;
        }
        return false;
    }

}
#endif
