#ifndef Kassiopeia_KSTrajTrajectoryMagneticBuilder_h_
#define Kassiopeia_KSTrajTrajectoryMagneticBuilder_h_

#include "KComplexElement.hh"
#include "KSTrajTrajectoryMagnetic.h"

using namespace Kassiopeia;
namespace katrin
{

    typedef KComplexElement< KSTrajTrajectoryMagnetic > KSTrajTrajectoryMagneticBuilder;

    template< >
    inline bool KSTrajTrajectoryMagneticBuilder::AddAttribute( KContainer* aContainer )
    {
        if( aContainer->GetName() == "name" )
        {
            aContainer->CopyTo( fObject, &KNamed::SetName );
            return true;
        }
        if( aContainer->GetName() == "reverse" )
        {
            aContainer->CopyTo( fObject, &KSTrajTrajectoryMagnetic::SetReverseDirection );
            return true;
        }
        return false;
    }

    template< >
    inline bool KSTrajTrajectoryMagneticBuilder::AddElement( KContainer* aContainer )
    {
        if( aContainer->Is< KSTrajMagneticIntegrator >() == true )
        {
            aContainer->ReleaseTo( fObject, &KSTrajTrajectoryMagnetic::SetIntegrator );
            return true;
        }
        if( aContainer->Is< KSTrajMagneticInterpolator >() == true )
        {
            aContainer->ReleaseTo( fObject, &KSTrajTrajectoryMagnetic::SetInterpolator );
            return true;
        }
        if( aContainer->Is< KSTrajMagneticDifferentiator >() == true )
        {
            aContainer->ReleaseTo( fObject, &KSTrajTrajectoryMagnetic::AddTerm );
            return true;
        }
        if( aContainer->Is< KSTrajMagneticControl >() == true )
        {
            aContainer->ReleaseTo( fObject, &KSTrajTrajectoryMagnetic::AddControl );
            return true;
        }
        return false;
    }

}



#endif
