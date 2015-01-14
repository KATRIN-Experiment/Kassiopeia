#ifndef Kassiopeia_KSRootTrajectoryBuilder_h_
#define Kassiopeia_KSRootTrajectoryBuilder_h_

#include "KComplexElement.hh"
#include "KSRootTrajectory.h"
#include "KSToolbox.h"

using namespace Kassiopeia;
namespace katrin
{

    typedef KComplexElement< KSRootTrajectory > KSRootTrajectoryBuilder;

    template< >
    inline bool KSRootTrajectoryBuilder::AddAttribute( KContainer* aContainer )
    {
        if( aContainer->GetName() == "name" )
        {
            aContainer->CopyTo( fObject, &KNamed::SetName );
            return true;
        }
        if( aContainer->GetName() == "set_trajectory" )
        {
            fObject->SetTrajectory( KSToolbox::GetInstance()->GetObjectAs< KSTrajectory >( aContainer->AsReference< string >() ) );
            return true;
        }
        return false;
    }

}

#endif
