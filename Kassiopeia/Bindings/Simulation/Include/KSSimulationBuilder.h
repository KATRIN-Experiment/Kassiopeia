#ifndef Kassiopeia_KSSimulationBuilder_h_
#define Kassiopeia_KSSimulationBuilder_h_

#include "KComplexElement.hh"
#include "KSSimulation.h"
#include "KSRootMagneticField.h"
#include "KSRootElectricField.h"
#include "KSRootSpace.h"
#include "KSRootGenerator.h"
#include "KSRootTrajectory.h"
#include "KSRootSpaceInteraction.h"
#include "KSRootSpaceNavigator.h"
#include "KSRootSurfaceInteraction.h"
#include "KSRootSurfaceNavigator.h"
#include "KSRootTerminator.h"
#include "KSRootWriter.h"
#include "KSToolbox.h"
#include "KSMainMessage.h"

using namespace Kassiopeia;
namespace katrin
{

    typedef KComplexElement< KSSimulation > KSSimulationBuilder;

    template< >
    inline bool KSSimulationBuilder::AddAttribute( KContainer* aContainer )
    {
        if( aContainer->GetName() == "name" )
        {
            aContainer->CopyTo( fObject, &KSSimulation::SetName );
            return true;
        }
        if( aContainer->GetName() == "seed" )
        {
            aContainer->CopyTo( fObject, &KSSimulation::SetSeed );
            return true;
        }
        if( aContainer->GetName() == "run" )
        {
            aContainer->CopyTo( fObject, &KSSimulation::SetRun );
            return true;
        }
        if( aContainer->GetName() == "events" )
        {
            aContainer->CopyTo( fObject, &KSSimulation::SetEvents );
            return true;
        }
        if( aContainer->GetName() == "step_report_iteration" )
        {
            if ( aContainer->AsReference<unsigned int>() == 0 )
            {
                mainmsg( eError ) << "Paramter of attribute <" <<"step_report_iteration" <<"> should not be 0"<<eom;
            }
            aContainer->CopyTo( fObject, &KSSimulation::SetStepReportIteration );
            return true;
        }
        if( aContainer->GetName() == "magnetic_field" )
        {
            KSComponent* tComponent = KSToolbox::GetInstance()->GetObjectAs< KSComponent >( aContainer->AsReference< string >() );
            KSCommand* tCommand = KSToolbox::GetInstance()->GetObjectAs< KSRootMagneticField >( "root_magnetic_field" )->Command( "add_magnetic_field", tComponent );
            fObject->AddCommand( tCommand );
            return true;
        }
        if( aContainer->GetName() == "electric_field" )
        {
            KSComponent* tComponent = KSToolbox::GetInstance()->GetObjectAs< KSComponent >( aContainer->AsReference< string >() );
            KSCommand* tCommand = KSToolbox::GetInstance()->GetObjectAs< KSRootElectricField >( "root_electric_field" )->Command( "add_electric_field", tComponent );
            fObject->AddCommand( tCommand );
            return true;
        }
        if( aContainer->GetName() == "space" )
        {
            KSComponent* tComponent = KSToolbox::GetInstance()->GetObjectAs< KSComponent >( aContainer->AsReference< string >() );
            KSCommand* tCommand = KSToolbox::GetInstance()->GetObjectAs< KSRootSpace >( "root_space" )->Command( "add_space", tComponent );
            fObject->AddCommand( tCommand );
            return true;
        }
        if( aContainer->GetName() == "surface" )
        {
            KSComponent* tComponent = KSToolbox::GetInstance()->GetObjectAs< KSComponent >( aContainer->AsReference< string >() );
            KSCommand* tCommand = KSToolbox::GetInstance()->GetObjectAs< KSRootSpace >( "root_space" )->Command( "add_surface", tComponent );
            fObject->AddCommand( tCommand );
            return true;
        }
        if( aContainer->GetName() == "generator" )
        {
            KSComponent* tComponent = KSToolbox::GetInstance()->GetObjectAs< KSComponent >( aContainer->AsReference< string >() );
            KSCommand* tCommand = KSToolbox::GetInstance()->GetObjectAs< KSRootGenerator >( "root_generator" )->Command( "set_generator", tComponent );
            fObject->AddCommand( tCommand );
            return true;
        }
        if( aContainer->GetName() == "trajectory" )
        {
            KSComponent* tComponent = KSToolbox::GetInstance()->GetObjectAs< KSComponent >( aContainer->AsReference< string >() );
            KSCommand* tCommand = KSToolbox::GetInstance()->GetObjectAs< KSRootTrajectory >( "root_trajectory" )->Command( "set_trajectory", tComponent );
            fObject->AddCommand( tCommand );
            return true;
        }
        if( aContainer->GetName() == "space_interaction" )
        {
            KSComponent* tComponent = KSToolbox::GetInstance()->GetObjectAs< KSComponent >( aContainer->AsReference< string >() );
            KSCommand* tCommand = KSToolbox::GetInstance()->GetObjectAs< KSRootSpaceInteraction >( "root_space_interaction" )->Command( "add_space_interaction", tComponent );
            fObject->AddCommand( tCommand );
            return true;
        }
        if( aContainer->GetName() == "space_navigator" )
        {
            KSComponent* tComponent = KSToolbox::GetInstance()->GetObjectAs< KSComponent >( aContainer->AsReference< string >() );
            KSCommand* tCommand = KSToolbox::GetInstance()->GetObjectAs< KSRootSpaceNavigator >( "root_space_navigator" )->Command( "set_space_navigator", tComponent );
            fObject->AddCommand( tCommand );
            return true;
        }
        if( aContainer->GetName() == "surface_interaction" )
        {
            KSComponent* tComponent = KSToolbox::GetInstance()->GetObjectAs< KSComponent >( aContainer->AsReference< string >() );
            KSCommand* tCommand = KSToolbox::GetInstance()->GetObjectAs< KSRootSurfaceInteraction >( "root_surface_interaction" )->Command( "set_surface_interaction", tComponent );
            fObject->AddCommand( tCommand );
            return true;
        }
        if( aContainer->GetName() == "surface_navigator" )
        {
            KSComponent* tComponent = KSToolbox::GetInstance()->GetObjectAs< KSComponent >( aContainer->AsReference< string >() );
            KSCommand* tCommand = KSToolbox::GetInstance()->GetObjectAs< KSRootSurfaceNavigator >( "root_surface_navigator" )->Command( "set_surface_navigator", tComponent );
            fObject->AddCommand( tCommand );
            return true;
        }
        if( aContainer->GetName() == "terminator" )
        {
            KSComponent* tComponent = KSToolbox::GetInstance()->GetObjectAs< KSComponent >( aContainer->AsReference< string >() );
            KSCommand* tCommand = KSToolbox::GetInstance()->GetObjectAs< KSRootTerminator >( "root_terminator" )->Command( "add_terminator", tComponent );
            fObject->AddCommand( tCommand );
            return true;
        }
        if( aContainer->GetName() == "writer" )
        {
            KSComponent* tComponent = KSToolbox::GetInstance()->GetObjectAs< KSComponent >( aContainer->AsReference< string >() );
            KSCommand* tCommand = KSToolbox::GetInstance()->GetObjectAs< KSRootWriter >( "root_writer" )->Command( "add_writer", tComponent );
            fObject->AddCommand( tCommand );
            return true;
        }
        if( aContainer->GetName() == "command" )
        {
            fObject->AddCommand( KSToolbox::GetInstance()->GetObjectAs< KSCommand >( aContainer->AsReference<string>() ));
            return true;
        }

        return false;
    }

}

#endif
