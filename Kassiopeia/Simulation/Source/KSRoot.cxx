#include "KSRoot.h"
#include "KSRunMessage.h"
#include "KSEventMessage.h"
#include "KSTrackMessage.h"
#include "KSStepMessage.h"

#include "KToolbox.h"
#include "KSNumerical.h"

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
#include "KSRootStepModifier.h"
#include "KSRootTrackModifier.h"
#include "KSRootEventModifier.h"
#include "KSRootRunModifier.h"

#include "KSParticle.h"
#include "KSParticleFactory.h"

#include "KSSimulation.h"
#include "KSRun.h"
#include "KSEvent.h"
#include "KSTrack.h"
#include "KSStep.h"

#include "KRandom.h"

#include <limits>
#include <signal.h>

using namespace std;
using namespace katrin;

namespace Kassiopeia
{
    bool KSRoot::fStopRunSignal   = false;
    bool KSRoot::fStopEventSignal = false;
    bool KSRoot::fStopTrackSignal = false;
    bool KSRoot::fGSLErrorSignal = false;
    string KSRoot::fGSLErrorString = "";
    string KSRoot::fStopSignalName = "";

    KSRoot::KSRoot() :
            fToolbox( KToolbox::GetInstance() ),
            fSimulation( nullptr ),
            fRun( nullptr ),
            fEvent( nullptr ),
            fTrack( nullptr ),
            fStep( nullptr ),
            fRootMagneticField( nullptr ),
            fRootElectricField( nullptr ),
            fRootSpace( nullptr ),
            fRootGenerator( nullptr ),
            fRootTrajectory( nullptr ),
            fRootSpaceInteraction( nullptr ),
            fRootSpaceNavigator( nullptr ),
            fRootSurfaceInteraction( nullptr ),
            fRootSurfaceNavigator( nullptr ),
            fRootTerminator( nullptr ),
            fRootWriter( nullptr ),
            fRootStepModifier( nullptr ),
            fRootTrackModifier( nullptr ),
            fRootEventModifier( nullptr ),
            fRootRunModifier( nullptr ),
            fRunIndex( 0 ),
            fEventIndex( 0 ),
            fTrackIndex( 0 ),
            fStepIndex( 0 )
    {
        fOnce = false;
        fRestartNavigation = true;

        this->SetName( "root" );

        if (fToolbox.Get<KSRun>("run") != NULL )
        {
            mainmsg(eWarning) << "New Kassiopeia instance will re-use already existing root objects." << eom;

            fRun   = fToolbox.Get<KSRun>("run");
            fEvent = fToolbox.Get<KSEvent>("event");
            fTrack = fToolbox.Get<KSTrack>("track");
            fStep  = fToolbox.Get<KSStep>("step");

            fRootMagneticField      = fToolbox.Get<KSRootMagneticField>("root_magnetic_field");
            fRootElectricField      = fToolbox.Get<KSRootElectricField>("root_electric_field");
            fRootSpace              = fToolbox.Get<KSRootSpace>("root_space");
            fRootGenerator          = fToolbox.Get<KSRootGenerator>("root_generator");
            fRootTrajectory         = fToolbox.Get<KSRootTrajectory>("root_trajectory");
            fRootSpaceInteraction   = fToolbox.Get<KSRootSpaceInteraction>("root_space_interaction");
            fRootSpaceNavigator     = fToolbox.Get<KSRootSpaceNavigator>("root_space_navigator");
            fRootSurfaceInteraction = fToolbox.Get<KSRootSurfaceInteraction>("root_surface_interaction");
            fRootSurfaceNavigator   = fToolbox.Get<KSRootSurfaceNavigator>("root_surface_navigator");
            fRootTerminator         = fToolbox.Get<KSRootTerminator>("root_terminator");
            fRootWriter             = fToolbox.Get<KSRootWriter>("root_writer");
            fRootStepModifier       = fToolbox.Get<KSRootStepModifier>("root_step_modifier");
            fRootTrackModifier      = fToolbox.Get<KSRootTrackModifier>("root_track_modifier");
            fRootEventModifier      = fToolbox.Get<KSRootEventModifier>("root_event_modifier");
            fRootRunModifier        = fToolbox.Get<KSRootRunModifier>("root_run_modifier");

            return;
        }

        fRun = new KSRun();
        fRun->SetName( "run" );
        fToolbox.Add<KSRun>(fRun, "run");

        fEvent = new KSEvent();
        fEvent->SetName( "event" );
        fToolbox.Add(fEvent);

        fTrack = new KSTrack();
        fTrack->SetName( "track" );
        fToolbox.Add(fTrack);

        fStep = new KSStep();
        fStep->SetName( "step" );
        fToolbox.Add(fStep);

        fRootMagneticField = new KSRootMagneticField();
        fRootMagneticField->SetName( "root_magnetic_field" );
        fToolbox.Add(fRootMagneticField);

        fRootElectricField = new KSRootElectricField();
        fRootElectricField->SetName( "root_electric_field" );
        fToolbox.Add(fRootElectricField);

        fRootSpace = new KSRootSpace();
        fRootSpace->SetName( "root_space" );
        fToolbox.Add(fRootSpace);

        fRootGenerator = new KSRootGenerator();
        fRootGenerator->SetName( "root_generator" );
        fRootGenerator->SetEvent( fEvent );
        fToolbox.Add(fRootGenerator);

        fRootTrajectory = new KSRootTrajectory();
        fRootTrajectory->SetName( "root_trajectory" );
        fRootTrajectory->SetStep( fStep );
        fToolbox.Add(fRootTrajectory);

        fRootSpaceInteraction = new KSRootSpaceInteraction();
        fRootSpaceInteraction->SetName( "root_space_interaction" );
        fRootSpaceInteraction->SetStep( fStep );
        fRootSpaceInteraction->SetTrajectory( fRootTrajectory );
        fToolbox.Add(fRootSpaceInteraction);

        fRootSpaceNavigator = new KSRootSpaceNavigator();
        fRootSpaceNavigator->SetName( "root_space_navigator" );
        fRootSpaceNavigator->SetStep( fStep );
        fRootSpaceNavigator->SetTrajectory( fRootTrajectory );
        fToolbox.Add(fRootSpaceNavigator);

        fRootSurfaceInteraction = new KSRootSurfaceInteraction();
        fRootSurfaceInteraction->SetName( "root_surface_interaction" );
        fRootSurfaceInteraction->SetStep( fStep );
        fToolbox.Add(fRootSurfaceInteraction);

        fRootSurfaceNavigator = new KSRootSurfaceNavigator();
        fRootSurfaceNavigator->SetName( "root_surface_navigator" );
        fRootSurfaceNavigator->SetStep( fStep );
        fToolbox.Add(fRootSurfaceNavigator);

        fRootTerminator = new KSRootTerminator();
        fRootTerminator->SetName( "root_terminator" );
        fRootTerminator->SetStep( fStep );
        fToolbox.Add(fRootTerminator);

        fRootWriter = new KSRootWriter();
        fRootWriter->SetName( "root_writer" );
        fToolbox.Add(fRootWriter);

        fRootStepModifier = new KSRootStepModifier();
        fRootStepModifier->SetName( "root_step_modifier" );
        fRootStepModifier->SetStep( fStep );
        fToolbox.Add( fRootStepModifier );

        fRootTrackModifier = new KSRootTrackModifier();
        fRootTrackModifier->SetName( "root_track_modifier" );
        fRootTrackModifier->SetTrack( fTrack );
        fToolbox.Add( fRootTrackModifier );

        fRootEventModifier = new KSRootEventModifier();
        fRootEventModifier->SetName( "root_event_modifier" );
        fRootEventModifier->SetEvent( fEvent );
        fToolbox.Add( fRootEventModifier );

        fRootRunModifier = new KSRootRunModifier();
        fRootRunModifier->SetName( "root_run_modifier" );
        fRootRunModifier->SetRun( fRun );
        fToolbox.Add( fRootRunModifier );

        KSParticleFactory::GetInstance().SetMagneticField( fRootMagneticField );
        KSParticleFactory::GetInstance().SetElectricField( fRootElectricField );
    }
    KSRoot::KSRoot( const KSRoot& aCopy) :
            KSComponent(),
            fToolbox( KToolbox::GetInstance() ),
            fSimulation( NULL ),
            fRun( aCopy.fRun ),
            fEvent( aCopy.fEvent ),
            fTrack( aCopy.fTrack ),
            fStep( aCopy.fStep ),
            fRootMagneticField( aCopy.fRootMagneticField ),
            fRootElectricField( aCopy.fRootElectricField ),
            fRootSpace( aCopy.fRootSpace ),
            fRootGenerator( aCopy.fRootGenerator ),
            fRootTrajectory( aCopy.fRootTrajectory ),
            fRootSpaceInteraction( aCopy.fRootSpaceInteraction ),
            fRootSpaceNavigator( aCopy.fRootSpaceNavigator ),
            fRootSurfaceInteraction( aCopy.fRootSurfaceInteraction ),
            fRootSurfaceNavigator( aCopy.fRootSurfaceNavigator ),
            fRootTerminator( aCopy.fRootTerminator ),
            fRootWriter( aCopy.fRootWriter ),
            fRootStepModifier( aCopy.fRootStepModifier ),
            fRootTrackModifier( aCopy.fRootTrackModifier ),
            fRootEventModifier( aCopy.fRootEventModifier ),
            fRootRunModifier( aCopy.fRootRunModifier ),
            fRunIndex( 0 ),
            fEventIndex( 0 ),
            fTrackIndex( 0 ),
            fStepIndex( 0 )
    {
        fOnce = false;
        fRestartNavigation = true;

        mainmsg(eWarning) << "Copied Kassiopeia instance will re-use already existing root objects." << eom;

        this->SetName( "root" );

        KSParticleFactory::GetInstance().SetMagneticField( fRootMagneticField );
        KSParticleFactory::GetInstance().SetElectricField( fRootElectricField );
    }
    KSRoot* KSRoot::Clone() const
    {
        return new KSRoot( *this );
    }
    KSRoot::~KSRoot()
    {
        /*
         * KToolbox takes care of destruction
         */
    }

    void KSRoot::Execute( KSSimulation* aSimulation )
    {
        if ( aSimulation != fSimulation )
        {
            fOnce = false;
        }

        fSimulation = aSimulation;

        vector< KSRunModifier* >* staticRunModifiers  = fSimulation->GetStaticRunModifiers();
        vector< KSEventModifier* >* staticEventModifiers = fSimulation->GetStaticEventModifiers();
        vector< KSTrackModifier* >* staticTrackModifiers = fSimulation->GetStaticTrackModifiers();
        vector< KSStepModifier* >* staticStepModifiers = fSimulation->GetStaticStepModifiers();

        if(!fOnce)
        {
            for(unsigned int i=0; i<staticRunModifiers->size(); i++)
            {
                staticRunModifiers->at(i)->Initialize();
                staticRunModifiers->at(i)->Activate();
                fRootRunModifier->AddModifier( staticRunModifiers->at(i) );
            };
            for(unsigned int i=0; i<staticEventModifiers->size(); i++)
            {
                staticEventModifiers->at(i)->Initialize();
                staticEventModifiers->at(i)->Activate();
                fRootEventModifier->AddModifier( staticEventModifiers->at(i) );
            };
            for(unsigned int i=0; i<staticTrackModifiers->size(); i++)
            {
                staticTrackModifiers->at(i)->Initialize();
                staticTrackModifiers->at(i)->Activate();
                fRootTrackModifier->AddModifier( staticTrackModifiers->at(i) );
            };
            for(unsigned int i=0; i<staticStepModifiers->size(); i++)
            {
                staticStepModifiers->at(i)->Initialize();
                staticStepModifiers->at(i)->Activate();
                fRootStepModifier->AddModifier( staticStepModifiers->at(i) );
            };

            Initialize();
            fSimulation->Initialize();

            Activate();
            fSimulation->Activate();

            fOnce = true;
        }

        //signal handling
        signal(SIGINT, &(KSRoot::SignalHandler) );
        signal(SIGTERM, &(KSRoot::SignalHandler) );
        signal(SIGQUIT, &(KSRoot::SignalHandler) );

        //GSL error handling
        fDefaultGSLErrorHandler = gsl_set_error_handler( &Kassiopeia::KSRoot::GSLErrorHandler );

        static const string sMessageSymbol = "\u263B ";
        mainmsg( eNormal ) << sMessageSymbol << "  welcome to Kassiopeia " << Kassiopeia_VERSION << "  " << sMessageSymbol << eom;  // version number from CMakeLists.txt

#ifdef Kassiopeia_ENABLE_DEBUG
        mainmsg( eWarning ) << "Kassiopeia is running in debug mode - compile without debug flags to speed up simulations." << eom;
#endif

        if (fSimulation->GetEvents() == 0)
        {
            mainmsg( eWarning ) << "Kassiopeia will not perform any tracking since the specified number of events is zero." << eom;
        }

        ExecuteRun();

        mainmsg( eNormal ) << "finished!" << eom;

        //reset GSL error handling
        if (fDefaultGSLErrorHandler != NULL)
        {
            gsl_set_error_handler( fDefaultGSLErrorHandler );
            fDefaultGSLErrorHandler = NULL;
        }

        //reset signal handling
        signal(SIGINT, SIG_DFL );
        signal(SIGTERM, SIG_DFL );
        signal(SIGQUIT, SIG_DFL );

        fSimulation->Deactivate();
        Deactivate();

        for(unsigned int i=0; i<staticRunModifiers->size(); i++)
        {
            staticRunModifiers->at(i)->Deactivate();
            staticRunModifiers->at(i)->Deinitialize();
        };
        for(unsigned int i=0; i<staticEventModifiers->size(); i++)
        {
            staticEventModifiers->at(i)->Deactivate();
            staticEventModifiers->at(i)->Deinitialize();

        };
        for(unsigned int i=0; i<staticTrackModifiers->size(); i++)
        {
            staticTrackModifiers->at(i)->Deactivate();
            staticTrackModifiers->at(i)->Deinitialize();
        };
        for(unsigned int i=0; i<staticStepModifiers->size(); i++)
        {
            staticStepModifiers->at(i)->Deactivate();
            staticStepModifiers->at(i)->Deinitialize();
        };


        fSimulation->Deinitialize();
        Deinitialize();
        fSimulation = NULL;

        fRunIndex = 0;
        fEventIndex = 0;
        fTrackIndex = 0;
        fStepIndex = 0;

        return;
    }

    void KSRoot::ExecuteRun()
    {
        // set random seed
        KRandom::GetInstance().SetSeed( fSimulation->GetSeed() );

        // reset run
        fRun->RunId() = fRunIndex;
        fRun->TotalEvents() = 0;
        fRun->TotalTracks() = 0;
        fRun->TotalSteps() = 0;
        fRun->ContinuousTime() = 0.;
        fRun->ContinuousLength() = 0.;
        fRun->ContinuousEnergyChange() = 0.;
        fRun->ContinuousMomentumChange() = 0.;
        fRun->DiscreteEnergyChange() = 0.;
        fRun->DiscreteMomentumChange() = 0.;
        fRun->DiscreteSecondaries() = 0;
        fRunIndex++;

        // send report
        runmsg( eNormal ) << "processing run " << fRun->GetRunId() << "..." << eom;

        while( true )
        {
            fRootRunModifier->ExecutePreRunModification();

            // break if done
            if( fRun->GetTotalEvents() == fSimulation->GetEvents() )
            {
                break;
            }

            //signal handler break
            if ( fStopRunSignal )
            {
                break;
            }

            // initialize event
            fEvent->ParentRunId() = fRun->GetRunId();

            // execute event
            ExecuteEvent();

            // update run
            fRun->TotalEvents() += 1;
            fRun->TotalTracks() += fEvent->TotalTracks();
            fRun->TotalSteps() += fEvent->TotalSteps();
            fRun->ContinuousTime() += fEvent->ContinuousTime();
            fRun->ContinuousLength() += fEvent->ContinuousLength();
            fRun->ContinuousEnergyChange() += fEvent->ContinuousEnergyChange();
            fRun->ContinuousMomentumChange() += fEvent->ContinuousMomentumChange();
            fRun->DiscreteEnergyChange() += fEvent->DiscreteEnergyChange();
            fRun->DiscreteMomentumChange() += fEvent->DiscreteMomentumChange();
            fRun->DiscreteSecondaries() += fEvent->DiscreteSecondaries();

            fRootRunModifier->ExecutePostRunModification();
        }

        // write run
        fRun->PushUpdate();
        fRootRunModifier->PushUpdate();

        fRootWriter->ExecuteRun();

        fRun->PushDeupdate();
        fRootRunModifier->PushDeupdate();

        // send report
        runmsg( eNormal ) << "...run " << fRun->GetRunId() << " complete" << eom;

        fStopRunSignal = false;
        return;
    }

    void KSRoot::ExecuteEvent()
    {
        // reset event
        fEvent->EventId() = fEventIndex;
        fEvent->TotalTracks() = 0;
        fEvent->TotalSteps() = 0;
        fEvent->ContinuousTime() = 0.;
        fEvent->ContinuousLength() = 0.;
        fEvent->ContinuousEnergyChange() = 0.;
        fEvent->ContinuousMomentumChange() = 0.;
        fEvent->DiscreteEnergyChange() = 0.;
        fEvent->DiscreteMomentumChange() = 0.;
        fEvent->DiscreteSecondaries() = 0;
        fEventIndex++;

        //clear any internal trajectory state
        fRootTrajectory->Reset();
        fRestartNavigation = true;

        fRootEventModifier->ExecutePreEventModification();

        // generate primaries
        fRootGenerator->ExecuteGeneration();

        // send report
        eventmsg( eNormal ) << "processing event " << fEvent->GetEventId() << " <" << fEvent->GetGeneratorName() << ">..." << eom;

        KSParticle* tParticle;
        while(!fEvent->ParticleQueue().empty())
        {
            //signal handler break
            if ( fStopRunSignal || fStopEventSignal )
            {
                //signal handler clears event queue
                while(!fEvent->ParticleQueue().empty())
                {
                    tParticle = fEvent->ParticleQueue().front();
                    delete tParticle;
                    fEvent->ParticleQueue().pop_front();
                }
                break;
            }

            // move the particle state to the track object
            tParticle = fEvent->ParticleQueue().front();
            tParticle->ReleaseLabel( fTrack->CreatorName() );
            fTrack->InitialParticle() = *tParticle;
            fTrack->FinalParticle() = *tParticle;

            // delete the particle and pop the queue
            delete tParticle;
            fEvent->ParticleQueue().pop_front();

            // execute a track
            ExecuteTrack();

            // move particles in track queue to event queue
            while(!fTrack->ParticleQueue().empty())
            {
                // pop a particle off the queue
                fEvent->ParticleQueue().push_back( fTrack->ParticleQueue().front() );
                fTrack->ParticleQueue().pop_front();
            }

            // update event
            fEvent->TotalTracks() += 1;
            fEvent->TotalSteps() += fTrack->GetTotalSteps();
            fEvent->ContinuousTime() += fTrack->ContinuousTime();
            fEvent->ContinuousLength() += fTrack->ContinuousLength();
            fEvent->ContinuousEnergyChange() += fTrack->ContinuousEnergyChange();
            fEvent->ContinuousMomentumChange() += fTrack->ContinuousMomentumChange();
            fEvent->DiscreteEnergyChange() += fTrack->DiscreteEnergyChange();
            fEvent->DiscreteMomentumChange() += fTrack->DiscreteMomentumChange();
            fEvent->DiscreteSecondaries() += fTrack->DiscreteSecondaries();
        }

        fRootEventModifier->ExecutePostEventModification();

        // write event
        fEvent->PushUpdate();
        fRootEventModifier->PushUpdate();

        fRootWriter->ExecuteEvent();

        fEvent->PushDeupdate();
        fRootEventModifier->PushDeupdate();

        // send report
        eventmsg( eNormal ) << "...completed event " << fEvent->GetEventId() << " <" << fEvent->GetGeneratorName() << ">" << eom;

        fStopEventSignal = false;
        return;
    }

    void KSRoot::ExecuteTrack()
    {
        // reset track
        fTrack->TrackId() = fTrackIndex;
        fTrack->TotalSteps() = 0;
        fTrack->ContinuousTime() = 0.;
        fTrack->ContinuousLength() = 0.;
        fTrack->ContinuousEnergyChange() = 0.;
        fTrack->ContinuousMomentumChange() = 0.;
        fTrack->DiscreteEnergyChange() = 0.;
        fTrack->DiscreteMomentumChange() = 0.;
        fTrack->DiscreteSecondaries() = 0;
        fStopTrackSignal = false;
        fGSLErrorSignal = false;

        // send report
        trackmsg( eNormal ) << "processing track " << fTrack->GetTrackId() << " <" << fTrack->GetCreatorName() << ">..." << eom;

        fRootTrackModifier->ExecutePreTrackModification();

        // start navigation
        if ( fRestartNavigation == true )
        {
            fRootSpaceNavigator->StartNavigation( fTrack->InitialParticle(), fRootSpace );
        }

        // initialize step objects
        fStep->InitialParticle() = fTrack->InitialParticle();
        fStep->FinalParticle() = fTrack->InitialParticle();

        //clear any internal trajectory state
        fRootTrajectory->Reset();
        fRestartNavigation = true;

        while( fStep->FinalParticle().IsActive() )
        {
            //signal handler break
            if ( fStopRunSignal || fStopEventSignal || fStopTrackSignal )
            {
                //signal handler clears event queue
                KSParticle* tParticle;
                while(!fTrack->ParticleQueue().empty())
                {
                    tParticle = fTrack->ParticleQueue().front();
                    delete tParticle;
                    fTrack->ParticleQueue().pop_front();
                }
                break;
            }

            // execute a step
            ExecuteStep();

            // move particles in step queue to track queue
            while(!fStep->ParticleQueue().empty())
            {
                // pop a particle off the queue
                fTrack->ParticleQueue().push_back( fStep->ParticleQueue().front() );
                fStep->ParticleQueue().pop_front();
            }

            if( fStep->TerminatorFlag() == false )
            {
                // update step objects
                fStep->InitialParticle() = fStep->FinalParticle();

                // update track
                fTrack->TotalSteps() += 1;
                fTrack->ContinuousTime() += fStep->ContinuousTime();
                fTrack->ContinuousLength() += fStep->ContinuousLength();
                fTrack->ContinuousEnergyChange() += fStep->ContinuousEnergyChange();
                fTrack->ContinuousMomentumChange() += fStep->ContinuousMomentumChange();
                fTrack->DiscreteEnergyChange() += fStep->DiscreteEnergyChange();
                fTrack->DiscreteMomentumChange() += fStep->DiscreteMomentumChange();
                fTrack->DiscreteSecondaries() += fStep->DiscreteSecondaries();
            }
        }

        fTrack->FinalParticle() = fStep->FinalParticle();
        fTrack->TerminatorName() = fStep->TerminatorName();

        fRootTrackModifier->ExecutePostTrackModification();


        // write track

        fTrack->PushUpdate();
        fRootTrackModifier->PushUpdate();

        fRootWriter->ExecuteTrack();

        fTrack->PushDeupdate();
        fRootTrackModifier->PushDeupdate();

        fTrackIndex++;

        // stop navigation
        if ( fRestartNavigation == true )
        {
            fRootSpaceNavigator->StopNavigation( fTrack->FinalParticle(), fRootSpace );
        };

        // send report
        trackmsg( eNormal ) << "...completed track " << fTrack->GetTrackId() << " <" << fTrack->GetTerminatorName() << "> after " << fTrack->GetTotalSteps() << " steps at " << fTrack->GetFinalParticle().GetPosition() << eom;

        fStopTrackSignal = false;
        fGSLErrorSignal = false;
        return;
    }

    void KSRoot::ExecuteStep()
    {
        // run pre-step modification
        bool hasPreModified = fRootStepModifier->ExecutePreStepModification();
        if (hasPreModified) {
            fRootTrajectory->Reset();
        }

        // reset step
        fStep->StepId() = fStepIndex;

        fStep->ContinuousTime() = 0.;
        fStep->ContinuousLength() = 0.;
        fStep->ContinuousEnergyChange() = 0.;
        fStep->ContinuousMomentumChange() = 0.;
        fStep->DiscreteSecondaries() = 0;
        fStep->DiscreteEnergyChange() = 0.;
        fStep->DiscreteMomentumChange() = 0.;

        fStep->TerminatorFlag() = false;
        fStep->TerminatorName().clear();

        fStep->TrajectoryName().clear();
        fStep->TrajectoryCenter().SetComponents( 0., 0., 0. );
        fStep->TrajectoryRadius() = 0.;
        fStep->TrajectoryStep() = numeric_limits< double >::max();

        fStep->SpaceInteractionName().clear();
        fStep->SpaceInteractionStep() = numeric_limits< double >::max();
        fStep->SpaceInteractionFlag() = false;

        fStep->SpaceNavigationName().clear();
        fStep->SpaceNavigationStep() = numeric_limits< double >::max();
        fStep->SpaceNavigationFlag() = false;

        fStep->SurfaceInteractionName().clear();
        fStep->SurfaceNavigationFlag() = false;

        fStep->SurfaceNavigationName().clear();
        fStep->SurfaceNavigationFlag() = false;

        // send report
        if( fStep->GetStepId() % fSimulation->GetStepReportIteration() == 0 )
        {
            stepmsg( eNormal ) << "processing step " << fStep->GetStepId() << "... (";
            stepmsg << "z = " << fStep->InitialParticle().GetPosition().Z() << ", ";
            stepmsg << "r = " << fStep->InitialParticle().GetPosition().Perp() << ", ";
            stepmsg << "k = " << fStep->InitialParticle().GetKineticEnergy_eV() << ", ";
            stepmsg << "e = " << fStep->InitialParticle().GetKineticEnergy_eV() + (fStep->InitialParticle().GetCharge() / KConst::Q()) * fStep->InitialParticle().GetElectricPotential();
            stepmsg << ")" << reom;
        }

        // debug spritz
        stepmsg_debug( "processing step " << fStep->GetStepId() << eom )
        stepmsg_debug( "step initial particle state: " << eom )
        stepmsg_debug( "  initial particle space: <" << (fStep->InitialParticle().GetCurrentSpace() ? fStep->InitialParticle().GetCurrentSpace()->GetName() : "" ) << ">" << eom )
        stepmsg_debug( "  initial particle surface: <" << (fStep->InitialParticle().GetCurrentSurface() ? fStep->InitialParticle().GetCurrentSurface()->GetName() : "" ) << ">" << eom )
        stepmsg_debug( "  initial particle side: <" << (fStep->InitialParticle().GetCurrentSide() ? fStep->InitialParticle().GetCurrentSide()->GetName() : "" ) << ">" << eom )
        stepmsg_debug( "  initial particle time: <" << fStep->InitialParticle().GetTime() << ">" << eom )
        stepmsg_debug( "  initial particle length: <" << fStep->InitialParticle().GetLength() << ">" << eom )
        stepmsg_debug( "  initial particle position: <" << fStep->InitialParticle().GetPosition().X() << ", " << fStep->InitialParticle().GetPosition().Y() << ", " << fStep->InitialParticle().GetPosition().Z() << ">" << eom )
        stepmsg_debug( "  initial particle momentum: <" << fStep->InitialParticle().GetMomentum().X() << ", " << fStep->InitialParticle().GetMomentum().Y() << ", " << fStep->InitialParticle().GetMomentum().Z() << ">" << eom )
        stepmsg_debug( "  initial particle kinetic energy: <" << fStep->InitialParticle().GetKineticEnergy_eV() << ">" << eom )
        stepmsg_debug( "  initial particle electric field: <" << fStep->InitialParticle().GetElectricField().X() << "," << fStep->InitialParticle().GetElectricField().Y() << "," << fStep->InitialParticle().GetElectricField().Z() << ">" << eom )
        stepmsg_debug( "  initial particle magnetic field: <" << fStep->InitialParticle().GetMagneticField().X() << "," << fStep->InitialParticle().GetMagneticField().Y() << "," << fStep->InitialParticle().GetMagneticField().Z() << ">" << eom )
        stepmsg_debug( "  initial particle angle to magnetic field: <" << fStep->InitialParticle().GetPolarAngleToB() << ">" << eom )

        //clear any abort signals in root trajectory
        KSTrajectory::ClearAbort();

        // run terminators
        fRootTerminator->CalculateTermination();

        // if terminators did not kill the particle, continue with calculations
        if( fStep->TerminatorFlag() == false )
        {
            // if the particle is not on a surface or side, continue with space calculations
            if( (fStep->InitialParticle().GetCurrentSurface() == nullptr) && (fStep->InitialParticle().GetCurrentSide() == nullptr) )
            {
                // integrate the trajectory
                fRootTrajectory->CalculateTrajectory();

                //need to check if an error handler event or stop signal
                //was triggered during the trajectory calculation
                if( fGSLErrorSignal || fStopTrackSignal || fRootTrajectory->Check() )
                {
                    fStep->FinalParticle().SetActive(false);
                    if( fRootTrajectory->Check() )
                    {
                        fStep->TerminatorName() = "trajectory_fail";
                    }
                    else
                    {
                        fStep->TerminatorName() = fStopSignalName;
                        if (fGSLErrorSignal)
                        {
                            mainmsg( eWarning ) << fGSLErrorString << eom;
                        }
                    }
                }
                else
                {
                    // calculate if a space interaction occurred
                    fRootSpaceInteraction->CalculateInteraction();

                    // calculate if a space navigation occurred
                    fRootSpaceNavigator->CalculateNavigation();

                    // if both a space interaction and space navigation occurred, differentiate between them based on which occurred first
                    if( (fStep->GetSpaceInteractionFlag() == true) && (fStep->GetSpaceNavigationFlag() == true) )
                    {

                        // if space interaction was first, execute it and clear space navigation data
                        if( fStep->GetSpaceInteractionStep() < fStep->GetSpaceNavigationStep() )
                        {
                            fRootSpaceInteraction->ExecuteInteraction();
                            fStep->SpaceNavigationName().clear();
                            fStep->SpaceNavigationStep() = numeric_limits< double >::max();
                            fStep->SpaceNavigationFlag() = false;

                            // if space interaction killed a particle, the terminator name is the space interaction name
                            if( fStep->FinalParticle().IsActive() == false )
                            {
                                fStep->TerminatorName() = fStep->SpaceInteractionName();
                            }
                        }
                        // if space navigation was first, execute it and clear space interaction data
                        else
                        {
                            fRootSpaceNavigator->ExecuteNavigation();
                            fStep->SpaceInteractionName().clear();
                            fStep->SpaceInteractionStep() = numeric_limits< double >::max();
                            fStep->SpaceInteractionFlag() = false;

                            // if space navigation killed a particle, the terminator name is the space navigation name
                            if( fStep->FinalParticle().IsActive() == false )
                            {
                                fStep->TerminatorName() = fStep->SpaceNavigationName();
                            }
                        }
                    }
                    // if only a space interaction occurred, execute it
                    else if( fStep->GetSpaceInteractionFlag() == true )
                    {
                        fRootSpaceInteraction->ExecuteInteraction();

                        // if space interaction killed a particle, the terminator name is the space interaction name
                        if( fStep->FinalParticle().IsActive() == false )
                        {
                            fStep->TerminatorName() = fStep->SpaceInteractionName();
                        }
                    }
                    // if only a space navigation occurred, execute it
                    else if( fStep->GetSpaceNavigationFlag() == true )
                    {
                        fRootSpaceNavigator->ExecuteNavigation();

                        // if space navigation killed a particle, the terminator name is the space navigation name
                        if( fStep->FinalParticle().IsActive() == false )
                        {
                            fStep->TerminatorName() = fStep->SpaceNavigationName();
                        }
                    }
                    // if neither occurred, execute the trajectory
                    else
                    {
                        fRootTrajectory->ExecuteTrajectory();
                    }

                    // execute post-step modification
                    bool hasPostModified = fRootStepModifier->ExecutePostStepModification();
                    if (hasPostModified)
                    {
                        fRootTrajectory->Reset();
                        if( fStep->FinalParticle().IsActive() == false )
                        {
                            fStep->TerminatorName() = fStep->ModifierName();
                            fRestartNavigation = false;  // this avoids losing internal navigation/terminator states
                        }
                    }

                    // push update
                    fStep->PushUpdate();
                    fRootTrajectory->PushUpdate();
                    fRootSpaceInteraction->PushUpdate();
                    fRootSpaceNavigator->PushUpdate();
                    fRootTerminator->PushUpdate();
                    fRootStepModifier->PushUpdate();

                    // write the step
                    fRootWriter->ExecuteStep();

                    // push deupdate
                    fStep->PushDeupdate();
                    fRootTrajectory->PushDeupdate();
                    fRootSpaceInteraction->PushDeupdate();
                    fRootSpaceNavigator->PushDeupdate();
                    fRootTerminator->PushDeupdate();
                    fRootStepModifier->PushDeupdate();
                }
            }
            // if the particle is on a surface or side, continue with surface calculations
            else
            {
                fRootSurfaceInteraction->ExecuteInteraction();

                // if surface interaction killed a particle, the terminator name is the surface interaction name
                if( fStep->InteractionParticle().IsActive() == false )
                {
                    fStep->TerminatorName() = fStep->SurfaceInteractionName();
                }

                fRootSurfaceNavigator->ExecuteNavigation();

                // if surface navigation killed a particle, the terminator name is the surface navigation name
                if( fStep->FinalParticle().IsActive() == false )
                {
                    fStep->TerminatorName() = fStep->SurfaceNavigationName();
                }

                // execute post-step modification
                bool hasPostModified = fRootStepModifier->ExecutePostStepModification();
                if (hasPostModified)
                {
                    fRootTrajectory->Reset();
                    if( fStep->FinalParticle().IsActive() == false )
                    {
                        fStep->TerminatorName() = fStep->ModifierName();
                        fRestartNavigation = false;  // this avoids losing internal navigation/terminator states
                    }
                }

                // push update
                fStep->PushUpdate();
                fRootSurfaceInteraction->PushUpdate();
                fRootSurfaceNavigator->PushUpdate();
                fRootTerminator->PushUpdate();
                fRootStepModifier->PushUpdate();

                // write the step
                fRootWriter->ExecuteStep();

                // push deupdate
                fStep->PushDeupdate();
                fRootSurfaceInteraction->PushDeupdate();
                fRootSurfaceNavigator->PushDeupdate();
                fRootTerminator->PushDeupdate();
                fRootStepModifier->PushDeupdate();
            }

            fStepIndex++;
        }
        // if the terminators killed the particle, execute them
        else
        {
            fRootTerminator->ExecuteTermination();
        }

        // label secondaries
        for( KSParticleIt tParticleIt = fStep->ParticleQueue().begin(); tParticleIt != fStep->ParticleQueue().end(); tParticleIt++ )
        {
            (*tParticleIt)->SetParentRunId( fRun->RunId() );
            (*tParticleIt)->SetParentEventId( fEvent->EventId() );
            (*tParticleIt)->SetParentTrackId( fTrack->TrackId() );
            (*tParticleIt)->SetParentStepId( fStep->StepId() );
        }

        stepmsg_debug( "finished step " << fStep->GetStepId() << eom )
        stepmsg_debug( "step final particle state: " << eom )
        stepmsg_debug( "  final particle space: <" << (fStep->FinalParticle().GetCurrentSpace() ? fStep->FinalParticle().GetCurrentSpace()->GetName() : "" ) << ">" << eom )
        stepmsg_debug( "  final particle surface: <" << (fStep->FinalParticle().GetCurrentSurface() ? fStep->FinalParticle().GetCurrentSurface()->GetName() : "" ) << ">" << eom )
        stepmsg_debug( "  final particle side: <" << (fStep->FinalParticle().GetCurrentSide() ? fStep->FinalParticle().GetCurrentSide()->GetName() : "" ) << ">" << eom )
        stepmsg_debug( "  final particle time: <" << fStep->FinalParticle().GetTime() << ">" << eom )
        stepmsg_debug( "  final particle length: <" << fStep->FinalParticle().GetLength() << ">" << eom )
        stepmsg_debug( "  final particle position: <" << fStep->FinalParticle().GetPosition().X() << ", " << fStep->FinalParticle().GetPosition().Y() << ", " << fStep->FinalParticle().GetPosition().Z() << ">" << eom )
        stepmsg_debug( "  final particle momentum: <" << fStep->FinalParticle().GetMomentum().X() << ", " << fStep->FinalParticle().GetMomentum().Y() << ", " << fStep->FinalParticle().GetMomentum().Z() << ">" << eom )
        stepmsg_debug( "  final particle kinetic energy: <" << fStep->FinalParticle().GetKineticEnergy_eV() << ">" << eom )
        stepmsg_debug( "  final particle electric field: <" << fStep->FinalParticle().GetElectricField().X() << "," << fStep->FinalParticle().GetElectricField().Y() << "," << fStep->FinalParticle().GetElectricField().Z() << ">" << eom )
        stepmsg_debug( "  final particle magnetic field: <" << fStep->FinalParticle().GetMagneticField().X() << "," << fStep->FinalParticle().GetMagneticField().Y() << "," << fStep->FinalParticle().GetMagneticField().Z() << ">" << eom )
        stepmsg_debug( "  final particle angle to magnetic field: <" << fStep->FinalParticle().GetPolarAngleToB() << ">" << eom )

        //signal handler terminate particle
        if ( fStopRunSignal || fStopEventSignal || fStopTrackSignal )
        {
            fStep->FinalParticle().SetActive( false );
            fStep->TerminatorName() = fStopSignalName;
        }

        //now that the step is completely done, finalize either the space or the surface navigation for the next step
        if ( fStep->FinalParticle().IsActive() == true )
        {
            if ( fStep->GetSpaceNavigationFlag() == true )
            {
                fRootSpaceNavigator->FinalizeNavigation();
            }
            else if ( fStep->GetSurfaceNavigationFlag() == true )
            {
                fRootSurfaceNavigator->FinalizeNavigation();
            }
        }

        return;
    }

    void KSRoot::InitializeComponent()
    {
        fRun->Initialize();
        fEvent->Initialize();
        fTrack->Initialize();
        fStep->Initialize();

        fRootMagneticField->Initialize();
        fRootElectricField->Initialize();
        fRootSpace->Initialize();
        fRootGenerator->Initialize();
        fRootTrajectory->Initialize();
        fRootSpaceInteraction->Initialize();
        fRootSpaceNavigator->Initialize();
        fRootSurfaceInteraction->Initialize();
        fRootSurfaceNavigator->Initialize();
        fRootTerminator->Initialize();
        fRootWriter->Initialize();
        fRootStepModifier->Initialize();
        fRootTrackModifier->Initialize();
        fRootEventModifier->Initialize();
        fRootRunModifier->Initialize();

        return;
    }

    void KSRoot::DeinitializeComponent()
    {
        fRun->Deinitialize();
        fEvent->Deinitialize();
        fTrack->Deinitialize();
        fStep->Deinitialize();

        fRootMagneticField->Deinitialize();
        fRootElectricField->Deinitialize();
        fRootGenerator->Deinitialize();
        fRootSpace->Deinitialize();
        fRootTrajectory->Deinitialize();
        fRootSpaceInteraction->Deinitialize();
        fRootSpaceNavigator->Deinitialize();
        fRootSurfaceInteraction->Deinitialize();
        fRootSurfaceNavigator->Deinitialize();
        fRootTerminator->Deinitialize();
        fRootWriter->Deinitialize();
        fRootStepModifier->Deinitialize();
        fRootTrackModifier->Deinitialize();
        fRootEventModifier->Deinitialize();
        fRootRunModifier->Deinitialize();

        return;
    }

    void KSRoot::ActivateComponent()
    {
        fRun->Activate();
        fEvent->Activate();
        fTrack->Activate();
        fStep->Activate();

        fRootMagneticField->Activate();
        fRootElectricField->Activate();
        fRootSpace->Activate();
        fRootGenerator->Activate();
        fRootTrajectory->Activate();
        fRootSpaceInteraction->Activate();
        fRootSpaceNavigator->Activate();
        fRootSurfaceInteraction->Activate();
        fRootSurfaceNavigator->Activate();
        fRootTerminator->Activate();
        fRootWriter->Activate();
        fRootStepModifier->Activate();
        fRootTrackModifier->Activate();
        fRootEventModifier->Activate();
        fRootRunModifier->Activate();

        return;
    }

    void KSRoot::DeactivateComponent()
    {
        fRun->Deactivate();
        fEvent->Deactivate();
        fTrack->Deactivate();
        fStep->Deactivate();

        fRootMagneticField->Deactivate();
        fRootElectricField->Deactivate();
        fRootGenerator->Deactivate();
        fRootSpace->Deactivate();
        fRootTrajectory->Deactivate();
        fRootSpaceInteraction->Deactivate();
        fRootSpaceNavigator->Deactivate();
        fRootSurfaceInteraction->Deactivate();
        fRootSurfaceNavigator->Deactivate();
        fRootTerminator->Deactivate();
        fRootWriter->Deactivate();
        fRootStepModifier->Deactivate();
        fRootTrackModifier->Deactivate();
        fRootEventModifier->Deactivate();
        fRootRunModifier->Deactivate();

        return;
    }

    void KSRoot::SignalHandler(int aSignal)
    {
        mainmsg( eWarning ) << "stop requested by signal <" << aSignal << ">. stopping simulation..." << eom;

        //set flag to stop run
        fStopRunSignal = true;
        fStopSignalName = "user_interrupt";

        // first signal stops tracking
        // second signal terminates the program immediately.
        signal(SIGINT, SIG_DFL);
        signal(SIGTERM, SIG_DFL);
        signal(SIGQUIT, SIG_DFL);
    }

    void KSRoot::GSLErrorHandler(const char* aReason, const char* aFile, int aLine, int aErrNo)
    {
        //cache the string for later since we may end up with many
        //gsl errors before the signal is caught and we dont want to clog up the terminal
        //only cache this message for the first error we encounter
        if(!fGSLErrorSignal)
        {
            stringstream ss;
            ss << "GSL error " << aErrNo << " <" << aReason << "> at <"  << aFile << ":" << aLine << ">. stopping track...";
            fGSLErrorString = ss.str();
        }

        //set flag to stop track
        fStopTrackSignal = true;
        fGSLErrorSignal = true;
        fStopSignalName = "gsl_error";

        KSTrajectory::SetAbort();
    }

}
