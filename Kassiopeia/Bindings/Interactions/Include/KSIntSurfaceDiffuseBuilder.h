#ifndef Kassiopeia_KSIntSurfaceDiffuseBuilder_h_
#define Kassiopeia_KSIntSurfaceDiffuseBuilder_h_

#include "KComplexElement.hh"
#include "KSIntSurfaceDiffuse.h"

using namespace Kassiopeia;
namespace katrin
{

    typedef KComplexElement< KSIntSurfaceDiffuse > KSIntSurfaceDiffuseBuilder;

    template< >
    inline bool KSIntSurfaceDiffuseBuilder::AddAttribute( KContainer* aContainer )
    {
        if( aContainer->GetName() == "name" )
        {
            aContainer->CopyTo( fObject, &KNamed::SetName );
            return true;
        }
        if( aContainer->GetName() == "probability" )
        {
            aContainer->CopyTo( fObject, &KSIntSurfaceDiffuse::SetProbability );
            return true;
        }
        if( aContainer->GetName() == "reflection_loss" )
        {
            aContainer->CopyTo( fObject, &KSIntSurfaceDiffuse::SetReflectionLoss );
            return true;
        }
        if( aContainer->GetName() == "transmission_loss" )
        {
            aContainer->CopyTo( fObject, &KSIntSurfaceDiffuse::SetTransmissionLoss );
            return true;
        }
        if( aContainer->GetName() == "reflection_loss_fraction" )
        {
            aContainer->CopyTo( fObject, &KSIntSurfaceDiffuse::SetReflectionLossFraction );
            return true;
        }
        if( aContainer->GetName() == "transmission_loss_fraction" )
        {
            aContainer->CopyTo( fObject, &KSIntSurfaceDiffuse::SetTransmissionLossFraction );
            return true;
        }
        return false;
    }

}

#endif
