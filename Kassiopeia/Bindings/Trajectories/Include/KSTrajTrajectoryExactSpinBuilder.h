#ifndef Kassiopeia_KSTrajTrajectoryExactSpinBuilder_h_
#define Kassiopeia_KSTrajTrajectoryExactSpinBuilder_h_

#include "KComplexElement.hh"
#include "KSTrajTrajectoryExactSpin.h"

using namespace Kassiopeia;
namespace katrin
{

    typedef KComplexElement< KSTrajTrajectoryExactSpin > KSTrajTrajectoryExactSpinBuilder;

    template< >
    inline bool KSTrajTrajectoryExactSpinBuilder::AddAttribute( KContainer* aContainer )
    {
        if( aContainer->GetName() == "name" )
        {
            aContainer->CopyTo( fObject, &KNamed::SetName );
            return true;
        }
        if( aContainer->GetName() == "piecewise_tolerance" )
        {
            aContainer->CopyTo( fObject, &KSTrajTrajectoryExactSpin::SetPiecewiseTolerance );
            return true;
        }
        if( aContainer->GetName() == "max_segments" )
        {
            aContainer->CopyTo( fObject, &KSTrajTrajectoryExactSpin::SetMaxNumberOfSegments );
            return true;
        }
        if( aContainer->GetName() == "attempt_limit" )
        {
            aContainer->CopyTo( fObject, &KSTrajTrajectoryExactSpin::SetAttemptLimit );
            return true;
        }
        return false;
    }

    template< >
    inline bool KSTrajTrajectoryExactSpinBuilder::AddElement( KContainer* aContainer )
    {
        if( aContainer->Is< KSTrajExactSpinIntegrator >() == true )
        {
            aContainer->ReleaseTo( fObject, &KSTrajTrajectoryExactSpin::SetIntegrator );
            return true;
        }
        if( aContainer->Is< KSTrajExactSpinInterpolator >() == true )
        {
            aContainer->ReleaseTo( fObject, &KSTrajTrajectoryExactSpin::SetInterpolator );
            return true;
        }
        if( aContainer->Is< KSTrajExactSpinDifferentiator >() == true )
        {
            aContainer->ReleaseTo( fObject, &KSTrajTrajectoryExactSpin::AddTerm );
            return true;
        }
        if( aContainer->Is< KSTrajExactSpinControl >() == true )
        {
            aContainer->ReleaseTo( fObject, &KSTrajTrajectoryExactSpin::AddControl );
            return true;
        }
        return false;
    }

}

#endif
