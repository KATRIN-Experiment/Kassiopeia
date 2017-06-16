#ifndef Kassiopeia_KSTrajTrajectoryExactTrappedBuilder_h_
#define Kassiopeia_KSTrajTrajectoryExactTrappedBuilder_h_

#include "KComplexElement.hh"
#include "KSTrajTrajectoryExactTrapped.h"

using namespace Kassiopeia;
namespace katrin
{

    typedef KComplexElement< KSTrajTrajectoryExactTrapped > KSTrajTrajectoryExactTrappedBuilder;

    template< >
    inline bool KSTrajTrajectoryExactTrappedBuilder::AddAttribute( KContainer* aContainer )
    {
        if( aContainer->GetName() == "name" )
        {
            aContainer->CopyTo( fObject, &KNamed::SetName );
            return true;
        }
        if( aContainer->GetName() == "piecewise_tolerance" )
        {
            aContainer->CopyTo( fObject, &KSTrajTrajectoryExactTrapped::SetPiecewiseTolerance );
            return true;
        }
        if( aContainer->GetName() == "max_segments" )
        {
            aContainer->CopyTo( fObject, &KSTrajTrajectoryExactTrapped::SetMaxNumberOfSegments );
            return true;
        }
        if( aContainer->GetName() == "attempt_limit" )
        {
            aContainer->CopyTo( fObject, &KSTrajTrajectoryExactTrapped::SetAttemptLimit );
            return true;
        }
        return false;
    }

    template< >
    inline bool KSTrajTrajectoryExactTrappedBuilder::AddElement( KContainer* aContainer )
    {
        if( aContainer->Is< KSTrajExactTrappedIntegrator >() == true )
        {
            aContainer->ReleaseTo( fObject, &KSTrajTrajectoryExactTrapped::SetIntegrator );
            return true;
        }
        //if( aContainer->Is< KSTrajExactTrappedInterpolator >() == true )
        //{
        //    aContainer->ReleaseTo( fObject, &KSTrajTrajectoryExactTrapped::SetInterpolator );
        //    return true;
        //}
        if( aContainer->Is< KSTrajExactTrappedDifferentiator >() == true )
        {
            aContainer->ReleaseTo( fObject, &KSTrajTrajectoryExactTrapped::AddTerm );
            return true;
        }
        if( aContainer->Is< KSTrajExactTrappedControl >() == true )
        {
            aContainer->ReleaseTo( fObject, &KSTrajTrajectoryExactTrapped::AddControl );
            return true;
        }
        return false;
    }

}

#endif
