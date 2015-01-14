#ifndef Kassiopeia_KSTrajTrajectoryLinearBuilder_h_
#define Kassiopeia_KSTrajTrajectoryLinearBuilder_h_

#include "KComplexElement.hh"
#include "KSTrajTrajectoryLinear.h"

using namespace Kassiopeia;
namespace katrin
{

    typedef KComplexElement< KSTrajTrajectoryLinear > KSTrajTrajectoryLinearBuilder;

    template< >
    inline bool KSTrajTrajectoryLinearBuilder::AddAttribute( KContainer* aContainer )
    {
        if( aContainer->GetName() == "name" )
        {
            aContainer->CopyTo( fObject, &KNamed::SetName );
            return true;
        }
        if( aContainer->GetName() == "length" )
        {
            aContainer->CopyTo( fObject, &KSTrajTrajectoryLinear::SetLength );
            return true;
        }
        return false;
    }

}

#endif
