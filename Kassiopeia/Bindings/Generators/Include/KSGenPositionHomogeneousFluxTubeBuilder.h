#ifndef KSGENPOSITIONHOMOGENEOUSFLUXTUBEBuilder_H_
#define KSGENPOSITIONHOMOGENEOUSFLUXTUBEBuilder_H_

#include "KSGenPositionHomogeneousFluxTube.h"
#include "KComplexElement.hh"
#include "KSToolbox.h"

using namespace Kassiopeia;
namespace katrin
{

    typedef KComplexElement< KSGenPositionHomogeneousFluxTube > KSGenPositionHomogeneousFluxTubeBuilder;

    template< >
    inline bool KSGenPositionHomogeneousFluxTubeBuilder::AddAttribute( KContainer* aContainer )
    {
        if( aContainer->GetName() == "name" )
        {
            aContainer->CopyTo( fObject, &KNamed::SetName );
            return true;
        }
        if( aContainer->GetName() == "phi_min" )
        {
        	aContainer->CopyTo( fObject, &KSGenPositionHomogeneousFluxTube::SetPhimin );
        	return true;
        }
        if( aContainer->GetName() == "phi_max" )
        {
        	aContainer->CopyTo( fObject, &KSGenPositionHomogeneousFluxTube::SetPhimax );
        	return true;
        }
        if( aContainer->GetName() == "z_min" )
        {
        	aContainer->CopyTo( fObject, &KSGenPositionHomogeneousFluxTube::SetZmin );
        	return true;
        }
        if( aContainer->GetName() == "z_max" )
        {
        	aContainer->CopyTo( fObject, &KSGenPositionHomogeneousFluxTube::SetZmax );
        	return true;
        }
        if( aContainer->GetName() == "flux" )
        {
            aContainer->CopyTo( fObject, &KSGenPositionHomogeneousFluxTube::SetFlux );
            return true;
        }
        if( aContainer->GetName() == "r_max" )
        {
            aContainer->CopyTo( fObject, &KSGenPositionHomogeneousFluxTube::SetRmax );
            return true;
        }
        if( aContainer->GetName() == "n_integration_step" )
        {
            aContainer->CopyTo( fObject, &KSGenPositionHomogeneousFluxTube::SetNIntegrationSteps );
            return true;
        }
        if( aContainer->GetName() == "magnetic_field_name" )
        {
            fObject->AddMagneticField( KSToolbox::GetInstance()->GetObjectAs< KSMagneticField >( aContainer->AsReference< string >() ) );
            return true;
        }
        return false;
    }

}

#endif /* KSGENPOSITIONHOMOGENEOUSFLUXTUBEBuilder_H_ */
