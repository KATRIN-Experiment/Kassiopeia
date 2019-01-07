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
        if( aContainer->GetName() == "eta" )
        {
            aContainer->CopyTo( fObject, &KSIntSurfaceUCN::SetEta );
            return true;
        }
        if( aContainer->GetName() == "alpha" )
        {
            aContainer->CopyTo( fObject, &KSIntSurfaceUCN::SetAlpha );
            return true;
        }
        if( aContainer->GetName() == "real_optical_potential" )
        {
            aContainer->CopyTo( fObject, &KSIntSurfaceUCN::SetRealOpticalPotential );
            return true;
        }
        if( aContainer->GetName() == "correlation_length" )
        {
            aContainer->CopyTo( fObject, &KSIntSurfaceUCN::SetCorrelationLength );
            return true;
        }
        return false;
    }

}
#endif
