#ifndef Kassiopeia_KSTrajTrajectoryExactBuilder_h_
#define Kassiopeia_KSTrajTrajectoryExactBuilder_h_

#include "KComplexElement.hh"
#include "KSTrajTrajectoryExact.h"

using namespace Kassiopeia;
namespace katrin
{

    typedef KComplexElement< KSTrajTrajectoryExact > KSTrajTrajectoryExactBuilder;

    template< >
    inline bool KSTrajTrajectoryExactBuilder::AddAttribute( KContainer* aContainer )
    {
        if( aContainer->GetName() == "name" )
        {
            aContainer->CopyTo( fObject, &KNamed::SetName );
            return true;
        }
        return false;
    }

    template< >
    inline bool KSTrajTrajectoryExactBuilder::AddElement( KContainer* aContainer )
    {
        if( aContainer->Is< KSTrajExactIntegrator >() == true )
        {
            aContainer->ReleaseTo( fObject, &KSTrajTrajectoryExact::SetIntegrator );
            return true;
        }
        if( aContainer->Is< KSTrajExactInterpolator >() == true )
        {
            aContainer->ReleaseTo( fObject, &KSTrajTrajectoryExact::SetInterpolator );
            return true;
        }
        if( aContainer->Is< KSTrajExactDifferentiator >() == true )
        {
            aContainer->ReleaseTo( fObject, &KSTrajTrajectoryExact::AddTerm );
            return true;
        }
        if( aContainer->Is< KSTrajExactControl >() == true )
        {
            aContainer->ReleaseTo( fObject, &KSTrajTrajectoryExact::AddControl );
            return true;
        }
        return false;
    }

}

#endif
