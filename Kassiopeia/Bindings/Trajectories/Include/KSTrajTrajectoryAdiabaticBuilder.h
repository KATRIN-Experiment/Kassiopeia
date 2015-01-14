#ifndef Kassiopeia_KSTrajTrajectoryAdiabaticBuilder_h_
#define Kassiopeia_KSTrajTrajectoryAdiabaticBuilder_h_

#include "KComplexElement.hh"
#include "KSTrajTrajectoryAdiabatic.h"

using namespace Kassiopeia;
namespace katrin
{

    typedef KComplexElement< KSTrajTrajectoryAdiabatic > KSTrajTrajectoryAdiabaticBuilder;

    template< >
    inline bool KSTrajTrajectoryAdiabaticBuilder::AddAttribute( KContainer* aContainer )
    {
        if( aContainer->GetName() == "name" )
        {
            aContainer->CopyTo( fObject, &KNamed::SetName );
            return true;
        }
        return false;
    }

    template< >
    inline bool KSTrajTrajectoryAdiabaticBuilder::AddElement( KContainer* aContainer )
    {
        if( aContainer->Is< KSTrajAdiabaticIntegrator >() == true )
        {
            aContainer->ReleaseTo( fObject, &KSTrajTrajectoryAdiabatic::SetIntegrator );
            return true;
        }
        if( aContainer->Is< KSTrajAdiabaticInterpolator >() == true )
        {
            aContainer->ReleaseTo( fObject, &KSTrajTrajectoryAdiabatic::SetInterpolator );
            return true;
        }
        if( aContainer->Is< KSTrajAdiabaticDifferentiator >() == true )
        {
            aContainer->ReleaseTo( fObject, &KSTrajTrajectoryAdiabatic::AddTerm );
            return true;
        }
        if( aContainer->Is< KSTrajAdiabaticControl >() == true )
        {
            aContainer->ReleaseTo( fObject, &KSTrajTrajectoryAdiabatic::AddControl );
            return true;
        }
        return false;
    }

}

#endif
