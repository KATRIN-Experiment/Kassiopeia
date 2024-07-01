#include "KSRoot.h"

#include "KElectricField.hh"
#include "KGslErrorHandler.h"
#include "KMagneticField.hh"
#include "KRandom.h"
#include "KSElectricKEMField.h"
#include "KSEvent.h"
#include "KSEventMessage.h"
#include "KSException.h"
#include "KSMagneticKEMField.h"
#include "KSNumerical.h"
#include "KSParticle.h"
#include "KSParticleFactory.h"
#include "KSRootElectricField.h"
#include "KSRootEventModifier.h"
#include "KSRootGenerator.h"
#include "KSRootMagneticField.h"
#include "KSRootRunModifier.h"
#include "KSRootSpace.h"
#include "KSRootSpaceInteraction.h"
#include "KSRootSpaceNavigator.h"
#include "KSRootStepModifier.h"
#include "KSRootSurfaceInteraction.h"
#include "KSRootSurfaceNavigator.h"
#include "KSRootTerminator.h"
#include "KSRootTrackModifier.h"
#include "KSRootTrajectory.h"
#include "KSRootWriter.h"
#include "KSRun.h"
#include "KSRunMessage.h"
#include "KSSimulation.h"
#include "KSStep.h"
#include "KSStepMessage.h"
#include "KSTrack.h"
#include "KSTrackMessage.h"
#include "KToolbox.h"

#include <chrono>
#include <csignal>
#include <limits>

using namespace std;
using namespace katrin;

namespace Kassiopeia
{
bool KSRoot::fStopRunSignal = false;
bool KSRoot::fStopEventSignal = false;
bool KSRoot::fStopTrackSignal = false;

KSRoot::KSRoot() :
    fToolbox(KToolbox::GetInstance()),
    fSimulation(nullptr),
    fRun(nullptr),
    fEvent(nullptr),
    fTrack(nullptr),
    fStep(nullptr),
    fRootMagneticField(nullptr),
    fRootElectricField(nullptr),
    fRootSpace(nullptr),
    fRootGenerator(nullptr),
    fRootTrajectory(nullptr),
    fRootSpaceInteraction(nullptr),
    fRootSpaceNavigator(nullptr),
    fRootSurfaceInteraction(nullptr),
    fRootSurfaceNavigator(nullptr),
    fRootTerminator(nullptr),
    fRootWriter(nullptr),
    fRootStepModifier(nullptr),
    fRootTrackModifier(nullptr),
    fRootEventModifier(nullptr),
    fRootRunModifier(nullptr),
    fRunIndex(0),
    fEventIndex(0),
    fTrackIndex(0),
    fStepIndex(0),
    fTotalExecTime(0)
{
    fOnce = false;
    fRestartNavigation = true;

    this->SetName("root");

    if (fToolbox.Get<KSRun>("run") != nullptr) {
        mainmsg(eWarning) << "New Kassiopeia instance will re-use already existing root objects." << eom;

        fRun = fToolbox.Get<KSRun>("run");
        fEvent = fToolbox.Get<KSEvent>("event");
        fTrack = fToolbox.Get<KSTrack>("track");
        fStep = fToolbox.Get<KSStep>("step");

        fRootMagneticField = fToolbox.Get<KSRootMagneticField>("root_magnetic_field");
        fRootElectricField = fToolbox.Get<KSRootElectricField>("root_electric_field");
        fRootSpace = fToolbox.Get<KSRootSpace>("root_space");
        fRootGenerator = fToolbox.Get<KSRootGenerator>("root_generator");
        fRootTrajectory = fToolbox.Get<KSRootTrajectory>("root_trajectory");
        fRootSpaceInteraction = fToolbox.Get<KSRootSpaceInteraction>("root_space_interaction");
        fRootSpaceNavigator = fToolbox.Get<KSRootSpaceNavigator>("root_space_navigator");
        fRootSurfaceInteraction = fToolbox.Get<KSRootSurfaceInteraction>("root_surface_interaction");
        fRootSurfaceNavigator = fToolbox.Get<KSRootSurfaceNavigator>("root_surface_navigator");
        fRootTerminator = fToolbox.Get<KSRootTerminator>("root_terminator");
        fRootWriter = fToolbox.Get<KSRootWriter>("root_writer");
        fRootStepModifier = fToolbox.Get<KSRootStepModifier>("root_step_modifier");
        fRootTrackModifier = fToolbox.Get<KSRootTrackModifier>("root_track_modifier");
        fRootEventModifier = fToolbox.Get<KSRootEventModifier>("root_event_modifier");
        fRootRunModifier = fToolbox.Get<KSRootRunModifier>("root_run_modifier");

        return;
    }

    fRun = new KSRun();
    fRun->SetName("run");
    fToolbox.Add<KSRun>(fRun, "run");

    fEvent = new KSEvent();
    fEvent->SetName("event");
    fToolbox.Add(fEvent);

    fTrack = new KSTrack();
    fTrack->SetName("track");
    fToolbox.Add(fTrack);

    fStep = new KSStep();
    fStep->SetName("step");
    fToolbox.Add(fStep);

    fRootMagneticField = new KSRootMagneticField();
    fRootMagneticField->SetName("root_magnetic_field");
    fToolbox.Add(fRootMagneticField);

    fRootElectricField = new KSRootElectricField();
    fRootElectricField->SetName("root_electric_field");
    fToolbox.Add(fRootElectricField);

    fRootSpace = new KSRootSpace();
    fRootSpace->SetName("root_space");
    fToolbox.Add(fRootSpace);

    fRootGenerator = new KSRootGenerator();
    fRootGenerator->SetName("root_generator");
    fRootGenerator->SetEvent(fEvent);
    fToolbox.Add(fRootGenerator);

    fRootTrajectory = new KSRootTrajectory();
    fRootTrajectory->SetName("root_trajectory");
    fRootTrajectory->SetStep(fStep);
    fToolbox.Add(fRootTrajectory);

    fRootSpaceInteraction = new KSRootSpaceInteraction();
    fRootSpaceInteraction->SetName("root_space_interaction");
    fRootSpaceInteraction->SetStep(fStep);
    fRootSpaceInteraction->SetTrajectory(fRootTrajectory);
    fToolbox.Add(fRootSpaceInteraction);

    fRootSpaceNavigator = new KSRootSpaceNavigator();
    fRootSpaceNavigator->SetName("root_space_navigator");
    fRootSpaceNavigator->SetStep(fStep);
    fRootSpaceNavigator->SetTrajectory(fRootTrajectory);
    fToolbox.Add(fRootSpaceNavigator);

    fRootSurfaceInteraction = new KSRootSurfaceInteraction();
    fRootSurfaceInteraction->SetName("root_surface_interaction");
    fRootSurfaceInteraction->SetStep(fStep);
    fToolbox.Add(fRootSurfaceInteraction);

    fRootSurfaceNavigator = new KSRootSurfaceNavigator();
    fRootSurfaceNavigator->SetName("root_surface_navigator");
    fRootSurfaceNavigator->SetStep(fStep);
    fToolbox.Add(fRootSurfaceNavigator);

    fRootTerminator = new KSRootTerminator();
    fRootTerminator->SetName("root_terminator");
    fRootTerminator->SetStep(fStep);
    fToolbox.Add(fRootTerminator);

    fRootWriter = new KSRootWriter();
    fRootWriter->SetName("root_writer");
    fToolbox.Add(fRootWriter);

    fRootStepModifier = new KSRootStepModifier();
    fRootStepModifier->SetName("root_step_modifier");
    fRootStepModifier->SetStep(fStep);
    fToolbox.Add(fRootStepModifier);

    fRootTrackModifier = new KSRootTrackModifier();
    fRootTrackModifier->SetName("root_track_modifier");
    fRootTrackModifier->SetTrack(fTrack);
    fToolbox.Add(fRootTrackModifier);

    fRootEventModifier = new KSRootEventModifier();
    fRootEventModifier->SetName("root_event_modifier");
    fRootEventModifier->SetEvent(fEvent);
    fToolbox.Add(fRootEventModifier);

    fRootRunModifier = new KSRootRunModifier();
    fRootRunModifier->SetName("root_run_modifier");
    fRootRunModifier->SetRun(fRun);
    fToolbox.Add(fRootRunModifier);

    // convert KEMField objects to Kassiopeia components
    for (auto& name : KToolbox::GetInstance().FindAll<KEMField::KElectricField>()) {
        auto object = KToolbox::GetInstance().Get<KEMField::KElectricField>(name);
        auto newObject = new KSElectricKEMField(object);
        newObject->Initialize();
        KToolbox::GetInstance().Add(newObject, name + "_");
    }
    for (auto& name : KToolbox::GetInstance().FindAll<KEMField::KMagneticField>()) {
        auto object = KToolbox::GetInstance().Get<KEMField::KMagneticField>(name);
        auto newObject = new KSMagneticKEMField(object);
        newObject->Initialize();
        KToolbox::GetInstance().Add(newObject, name + "_");
    }
}
KSRoot::KSRoot(const KSRoot& aCopy) :
    KSComponent(aCopy),
    fToolbox(KToolbox::GetInstance()),
    fSimulation(nullptr),
    fRun(aCopy.fRun),
    fEvent(aCopy.fEvent),
    fTrack(aCopy.fTrack),
    fStep(aCopy.fStep),
    fRootMagneticField(aCopy.fRootMagneticField),
    fRootElectricField(aCopy.fRootElectricField),
    fRootSpace(aCopy.fRootSpace),
    fRootGenerator(aCopy.fRootGenerator),
    fRootTrajectory(aCopy.fRootTrajectory),
    fRootSpaceInteraction(aCopy.fRootSpaceInteraction),
    fRootSpaceNavigator(aCopy.fRootSpaceNavigator),
    fRootSurfaceInteraction(aCopy.fRootSurfaceInteraction),
    fRootSurfaceNavigator(aCopy.fRootSurfaceNavigator),
    fRootTerminator(aCopy.fRootTerminator),
    fRootWriter(aCopy.fRootWriter),
    fRootStepModifier(aCopy.fRootStepModifier),
    fRootTrackModifier(aCopy.fRootTrackModifier),
    fRootEventModifier(aCopy.fRootEventModifier),
    fRootRunModifier(aCopy.fRootRunModifier),
    fRunIndex(0),
    fEventIndex(0),
    fTrackIndex(0),
    fStepIndex(0),
    fTotalExecTime(0)
{
    fOnce = false;
    fRestartNavigation = true;

    mainmsg(eWarning) << "Copied Kassiopeia instance will re-use already existing root objects." << eom;

    this->SetName("root");
}
KSRoot* KSRoot::Clone() const
{
    return new KSRoot(*this);
}
KSRoot::~KSRoot() = default;
//{
/*
 * KToolbox takes care of destruction
 */
//}

void KSRoot::Execute(KSSimulation* aSimulation)
{
    if (aSimulation != fSimulation) {
        fOnce = false;
    }

    fSimulation = aSimulation;

    fRunIndex = 0;
    fEventIndex = 0;
    fTrackIndex = 0;
    fStepIndex = 0;

    fStopRunSignal = false;
    fStopEventSignal = false;
    fStopTrackSignal = false;

    fTotalExecTime = 0;

    vector<KSRunModifier*>* staticRunModifiers = fSimulation->GetStaticRunModifiers();
    vector<KSEventModifier*>* staticEventModifiers = fSimulation->GetStaticEventModifiers();
    vector<KSTrackModifier*>* staticTrackModifiers = fSimulation->GetStaticTrackModifiers();
    vector<KSStepModifier*>* staticStepModifiers = fSimulation->GetStaticStepModifiers();

    if (!fOnce) {
        Initialize();
        fSimulation->Initialize();

        for (auto& staticRunModifier : *staticRunModifiers) {
            staticRunModifier->Initialize();
            staticRunModifier->Activate();
            fRootRunModifier->AddModifier(staticRunModifier);
        };
        for (auto& staticEventModifier : *staticEventModifiers) {
            staticEventModifier->Initialize();
            staticEventModifier->Activate();
            fRootEventModifier->AddModifier(staticEventModifier);
        };
        for (auto& staticTrackModifier : *staticTrackModifiers) {
            staticTrackModifier->Initialize();
            staticTrackModifier->Activate();
            fRootTrackModifier->AddModifier(staticTrackModifier);
        };
        for (auto& staticStepModifier : *staticStepModifiers) {
            staticStepModifier->Initialize();
            staticStepModifier->Activate();
            fRootStepModifier->AddModifier(staticStepModifier);
        };

        Activate();
        fSimulation->Activate();

        fOnce = true;
    }

    KSParticleFactory::GetInstance().SetMagneticField(fRootMagneticField);
    KSParticleFactory::GetInstance().SetElectricField(fRootElectricField);

    static const string sMessageSymbol = "\u263B ";
    mainmsg(eNormal) << sMessageSymbol << "  welcome to Kassiopeia " << Kassiopeia_VERSION << "  " << sMessageSymbol
                     << eom;  // version number from CMakeLists.txt

#ifdef Kassiopeia_ENABLE_DEBUG
    mainmsg(eWarning) << "Kassiopeia is running in debug mode - compile without debug flags to speed up simulations."
                      << eom;
#endif

    if (fSimulation->GetEvents() == 0) {
        mainmsg(eWarning) << "Kassiopeia will not perform any tracking since the specified number of events is zero."
                          << eom;
    }
    else {
        if (fSimulation->GetEvents() > 1000000)  // can happen if passing events=-1 to the builder
        {
            mainmsg(eWarning) << "Kassiopeia will simulate <" << fSimulation->GetEvents() << "> events - that's a lot!"
                              << eom;
        }

        //signal handling
        signal(SIGINT, &(KSRoot::SignalHandler));
        signal(SIGTERM, &(KSRoot::SignalHandler));
        signal(SIGQUIT, &(KSRoot::SignalHandler));

        //enable GSL error handling
        KGslErrorHandler::GetInstance().Reset();
        KGslErrorHandler::GetInstance().Enable();

        try {
            ExecuteRun();
        }
        catch (KSUserInterrupt const& e) {
            stepmsg(eInfo) << "Interrupted simulation (" << e.what() << ")" << eom;
            // do nothing (run already stopped)
        }
        catch (KException const& e) {
            stepmsg(eWarning) << "Failed to execute simulation (" << e.what() << ")" << eom;
            // do nothing
        }

        mainmsg(eNormal) << "finished!" << eom;

        //reset GSL error handling
        KGslErrorHandler::GetInstance().Reset();

        //reset signal handling
        signal(SIGINT, SIG_DFL);
        signal(SIGTERM, SIG_DFL);
        signal(SIGQUIT, SIG_DFL);
    }

    fSimulation->Deactivate();
    Deactivate();

    for (auto& staticRunModifier : *staticRunModifiers) {
        staticRunModifier->Deactivate();
        staticRunModifier->Deinitialize();
    };
    for (auto& staticEventModifier : *staticEventModifiers) {
        staticEventModifier->Deactivate();
        staticEventModifier->Deinitialize();
    };
    for (auto& staticTrackModifier : *staticTrackModifiers) {
        staticTrackModifier->Deactivate();
        staticTrackModifier->Deinitialize();
    };
    for (auto& staticStepModifier : *staticStepModifiers) {
        staticStepModifier->Deactivate();
        staticStepModifier->Deinitialize();
    };

    fSimulation->Deinitialize();
    Deinitialize();

    return;
}

void KSRoot::ExecuteRun()
{
    // set random seed
    KRandom::GetInstance().SetSeed(fSimulation->GetSeed());

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
    fRun->NumberOfTurns() = 0;
    fRunIndex++;

    //clear any previous GSL errors
    KGslErrorHandler::GetInstance().ClearError();

    fRun->StartTiming();

    // send report
    runmsg(eNormal) << "processing run " << fRun->GetRunId() << " ..." << eom;

    while (true) {
        fRootRunModifier->ExecutePreRunModification();

        // break if done
        if (fRun->GetTotalEvents() >= fSimulation->GetEvents()) {
            break;
        }

        //signal handler break
        if (fStopRunSignal) {
            break;
        }

        // initialize event
        fEvent->ParentRunId() = fRun->GetRunId();

        // execute event
        try {
            ExecuteEvent();
        }
        catch (KSUserInterrupt const& e) {
            stepmsg(eInfo) << "Interrupted at run <" << fRun->RunId() << "> (" << e.what() << ")" << eom;
            // stop current run
            fStopRunSignal = true;
        }
        catch (KException const& e) {
            stepmsg(eWarning) << "Failed to execute run <" << fRun->RunId() << "> (" << e.what() << ")" << eom;
            // stop current run
            fStopRunSignal = true;
        }

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
        fRun->NumberOfTurns() += fEvent->NumberOfTurns();

        fRootRunModifier->ExecutePostRunModification();
    }

    fRun->EndTiming();

    // write run
    fRun->PushUpdate();
    fRootRunModifier->PushUpdate();

    fRootWriter->ExecuteRun();

    fRun->PushDeupdate();
    fRootRunModifier->PushDeupdate();

    // send report
    runmsg(eNormal) << "...run " << fRun->GetRunId() << " complete" << eom;

    fStopRunSignal = false;
    KGslErrorHandler::GetInstance().ClearError();
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
    fEvent->NumberOfTurns() = 0;
    fEventIndex++;

    fEvent->StartTiming();

    fRootEventModifier->ExecutePreEventModification();

    // generate primaries
    fRootGenerator->ExecuteGeneration();

    // send report
    eventmsg(eNormal) << "processing event " << fEvent->GetEventId() << " <" << fEvent->GetGeneratorName() << "> ..."
                      << eom;

    //clear any internal trajectory state
    fRootTrajectory->Reset();
    fRestartNavigation = true;

    //clear any previous GSL errors
    KGslErrorHandler::GetInstance().ClearError();

    KSParticle* tParticle;
    while (!fEvent->ParticleQueue().empty()) {
        //signal handler break
        if (fStopRunSignal || fStopEventSignal) {
            //signal handler clears event queue
            while (!fEvent->ParticleQueue().empty()) {
                tParticle = fEvent->ParticleQueue().front();
                delete tParticle;
                fEvent->ParticleQueue().pop_front();
            }
            break;
        }

        // move the particle state to the track object
        tParticle = fEvent->ParticleQueue().front();
        tParticle->ReleaseLabel(fTrack->CreatorName());
        fTrack->InitialParticle() = *tParticle;
        fTrack->FinalParticle() = *tParticle;

        // delete the particle and pop the queue
        delete tParticle;
        fEvent->ParticleQueue().pop_front();

        // execute a track
        try {
            ExecuteTrack();
        }
        catch (KSUserInterrupt const& e) {
            stepmsg(eInfo) << "Interrupted at event <" << fEvent->EventId() << "> (" << e.what() << ")" << eom;
            // stop current run
            fStopRunSignal = true;
        }
        catch (KException const& e) {
            stepmsg(eWarning) << "Failed to execute event <" << fEvent->EventId() << "> (" << e.what() << ")" << eom;
            // stop current event
            fStopEventSignal = true;
        }

        // move particles in track queue to event queue
        while (!fTrack->ParticleQueue().empty()) {
            // pop a particle off the queue
            fEvent->ParticleQueue().push_back(fTrack->ParticleQueue().front());
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
        fEvent->NumberOfTurns() += fTrack->NumberOfTurns();
    }

    fRootEventModifier->ExecutePostEventModification();

    fEvent->EndTiming();

    // write event
    fEvent->PushUpdate();
    fRootEventModifier->PushUpdate();

    fRootWriter->ExecuteEvent();

    fEvent->PushDeupdate();
    fRootEventModifier->PushDeupdate();

    // determine time spent for event processing
    auto tTimeSpan = fEvent->GetProcessingDuration();

    fTotalExecTime += tTimeSpan;

    auto tEventsDone = fRun->GetTotalEvents() + 1;
    auto tEventsLeft = fSimulation->GetEvents() - tEventsDone;

    auto tTimePerEvent = fTotalExecTime / tEventsDone;
    auto tTimeLeft = tTimePerEvent * tEventsLeft;

    // send report
    eventmsg(eNormal) << "...completed event " << fEvent->GetEventId() << " <" << fEvent->GetGeneratorName() << "> ";
    eventmsg(eNormal) << " in " << ceil(tTimeSpan) << " s (ca. " << ceil(tTimeLeft) << " s left)" << eom;

    fStopEventSignal = false;
    KGslErrorHandler::GetInstance().ClearError();
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
    fTrack->NumberOfTurns() = 0;

    fTrack->StartTiming();

    fRootTrackModifier->ExecutePreTrackModification();

    // send report
    trackmsg(eNormal) << "processing track " << fTrack->GetTrackId() << " <" << fTrack->GetCreatorName() << "> ..."
                      << eom;

    // start navigation
    if (fRestartNavigation == true) {
        fRootSpaceNavigator->StartNavigation(fTrack->InitialParticle(), fRootSpace);
    }

    // initialize step objects
    fStep->InitialParticle() = fTrack->InitialParticle();
    fStep->FinalParticle() = fTrack->InitialParticle();

    //clear any internal trajectory state
    fRootTrajectory->Reset();
    fRestartNavigation = true;

    //clear any previous GSL errors
    KGslErrorHandler::GetInstance().ClearError();

    while (fStep->FinalParticle().IsActive()) {
        //signal handler break
        if (fStopRunSignal || fStopEventSignal || fStopTrackSignal) {
            //signal handler clears event queue
            KSParticle* tParticle;
            while (!fTrack->ParticleQueue().empty()) {
                tParticle = fTrack->ParticleQueue().front();
                delete tParticle;
                fTrack->ParticleQueue().pop_front();
            }
            break;
        }

        // execute a step
        try {
            ExecuteStep();
        }
        catch (KSUserInterrupt const& e) {
            stepmsg(eInfo) << "Interrupted at track <" << fTrack->TrackId() << "> (" << e.what() << ")" << eom;
            // stop current run
            fStopRunSignal = true;
        }
        catch (KException const& e) {
            stepmsg(eWarning) << "Failed to execute track <" << fTrack->TrackId() << "> (" << e.what() << ")" << eom;
            // stop current track
            fStopTrackSignal = true;
        }

        // move particles in step queue to track queue
        while (!fStep->ParticleQueue().empty()) {
            // pop a particle off the queue
            fTrack->ParticleQueue().push_back(fStep->ParticleQueue().front());
            fStep->ParticleQueue().pop_front();
        }

        if (fStep->TerminatorFlag() == false) {
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
            fTrack->NumberOfTurns() += fStep->NumberOfTurns();
        }
    }

    fTrack->FinalParticle() = fStep->FinalParticle();
    fTrack->TerminatorName() = fStep->TerminatorName();

    fRootTrackModifier->ExecutePostTrackModification();

    fTrack->EndTiming();


    // write track

    fTrack->PushUpdate();
    fRootTrackModifier->PushUpdate();

    fRootWriter->ExecuteTrack();

    fTrack->PushDeupdate();
    fRootTrackModifier->PushDeupdate();

    fTrackIndex++;

    // stop navigation
    if (fRestartNavigation == true) {
        fRootSpaceNavigator->StopNavigation(fTrack->FinalParticle(), fRootSpace);
    };

    // send report
    trackmsg(eNormal) << "...completed track " << fTrack->GetTrackId() << " <" << fTrack->GetTerminatorName() << "> ";
    trackmsg(eNormal) << "after " << fTrack->GetTotalSteps() << " steps at " << fTrack->GetFinalParticle().GetPosition()
                      << eom;

    fStopTrackSignal = false;
    KGslErrorHandler::GetInstance().ClearError();
    return;
}

void KSRoot::ExecuteStep()
{
    // reset step
    fStep->StepId() = fStepIndex;

    fStep->ContinuousTime() = 0.;
    fStep->ContinuousLength() = 0.;
    fStep->ContinuousEnergyChange() = 0.;
    fStep->ContinuousMomentumChange() = 0.;
    fStep->DiscreteSecondaries() = 0;
    fStep->DiscreteEnergyChange() = 0.;
    fStep->DiscreteMomentumChange() = 0.;
    fStep->NumberOfTurns() = 0;

    fStep->TerminatorFlag() = false;
    fStep->TerminatorName().clear();

    fStep->TrajectoryName().clear();
    fStep->TrajectoryCenter().SetComponents(0., 0., 0.);
    fStep->TrajectoryRadius() = 0.;
    fStep->TrajectoryStep() = numeric_limits<double>::max();

    fStep->SpaceInteractionName().clear();
    fStep->SpaceInteractionStep() = numeric_limits<double>::max();
    fStep->SpaceInteractionFlag() = false;

    fStep->SpaceNavigationName().clear();
    fStep->SpaceNavigationStep() = numeric_limits<double>::max();
    fStep->SpaceNavigationFlag() = false;

    fStep->SurfaceInteractionName().clear();
    fStep->SurfaceNavigationFlag() = false;

    fStep->SurfaceNavigationName().clear();
    fStep->SurfaceNavigationFlag() = false;

    fStep->StartTiming();

    // run pre-step modification
    bool hasPreModified = fRootStepModifier->ExecutePreStepModification();
    if (hasPreModified) {
        fRootTrajectory->Reset();
    }

    // send report
    if (fStep->GetStepId() % fSimulation->GetStepReportIteration() == 0) {
        stepmsg(eNormal) << "processing step " << fStep->GetStepId() << " ... (";
        stepmsg << "z = " << fStep->InitialParticle().GetPosition().Z() << ", ";
        stepmsg << "r = " << fStep->InitialParticle().GetPosition().Perp() << ", ";
        stepmsg << "k = " << fStep->InitialParticle().GetKineticEnergy_eV() << ", ";
        stepmsg << "e = "
                << fStep->InitialParticle().GetKineticEnergy_eV() +
                       (fStep->InitialParticle().GetCharge() / katrin::KConst::Q()) *
                           fStep->InitialParticle().GetElectricPotential();
        stepmsg << ")" << reom;
    }

    // debug spritz
    stepmsg_debug("processing step " << fStep->GetStepId() << eom);
    stepmsg_debug("step initial particle state: " << eom);
    stepmsg_debug("  initial particle space: <"
                  << (fStep->InitialParticle().GetCurrentSpace() ? fStep->InitialParticle().GetCurrentSpace()->GetName()
                                                                 : "")
                  << ">" << eom);
    stepmsg_debug("  initial particle surface: <" << (fStep->InitialParticle().GetCurrentSurface()
                                                          ? fStep->InitialParticle().GetCurrentSurface()->GetName()
                                                          : "")
                                                  << ">" << eom);
    stepmsg_debug("  initial particle side: <"
                  << (fStep->InitialParticle().GetCurrentSide() ? fStep->InitialParticle().GetCurrentSide()->GetName()
                                                                : "")
                  << ">" << eom);
    stepmsg_debug("  initial particle time: <" << fStep->InitialParticle().GetTime() << ">" << eom);
    stepmsg_debug("  initial particle length: <" << fStep->InitialParticle().GetLength() << ">" << eom);
    stepmsg_debug("  initial particle position: <" << fStep->InitialParticle().GetPosition().X() << ", "
                                                   << fStep->InitialParticle().GetPosition().Y() << ", "
                                                   << fStep->InitialParticle().GetPosition().Z() << ">" << eom);
    stepmsg_debug("  initial particle momentum: <" << fStep->InitialParticle().GetMomentum().X() << ", "
                                                   << fStep->InitialParticle().GetMomentum().Y() << ", "
                                                   << fStep->InitialParticle().GetMomentum().Z() << ">" << eom);
    stepmsg_debug("  initial particle kinetic energy: <" << fStep->InitialParticle().GetKineticEnergy_eV() << ">"
                                                         << eom);
    stepmsg_debug("  initial particle electric field: <" << fStep->InitialParticle().GetElectricField().X() << ","
                                                         << fStep->InitialParticle().GetElectricField().Y() << ","
                                                         << fStep->InitialParticle().GetElectricField().Z() << ">"
                                                         << eom);
    stepmsg_debug("  initial particle magnetic field: <" << fStep->InitialParticle().GetMagneticField().X() << ","
                                                         << fStep->InitialParticle().GetMagneticField().Y() << ","
                                                         << fStep->InitialParticle().GetMagneticField().Z() << ">"
                                                         << eom);
    stepmsg_debug("  initial particle angle to magnetic field: <" << fStep->InitialParticle().GetPolarAngleToB() << ">"
                                                                  << eom);
    stepmsg_debug("  initial particle spin: " << fStep->InitialParticle().GetSpin() << eom);
    stepmsg_debug("  initial particle spin0: <" << fStep->InitialParticle().GetSpin0() << ">" << eom);
    stepmsg_debug("  initial particle aligned spin: <" << fStep->InitialParticle().GetAlignedSpin() << ">" << eom);
    stepmsg_debug("  initial particle spin angle: <" << fStep->InitialParticle().GetSpinAngle() << ">" << eom);

    //clear any abort signals in root trajectory
    KSTrajectory::ClearAbort();

    //clear any previous GSL errors
    KGslErrorHandler::GetInstance().ClearError();

    try {
        // run terminators
        fRootTerminator->CalculateTermination();

        // if terminators did not kill the particle, continue with calculations
        if (fStep->TerminatorFlag() == false) {
            // if the particle is not on a surface or side, continue with space calculations
            if ((fStep->InitialParticle().GetCurrentSurface() == nullptr) &&
                (fStep->InitialParticle().GetCurrentSide() == nullptr)) {
                // integrate the trajectory
                fRootTrajectory->CalculateTrajectory();

                //need to check if an error handler event or stop signal
                //was triggered during the trajectory calculation
                if (fStopTrackSignal) {
                    fStep->FinalParticle().SetActive(false);
                }
                else {
                    // calculate if a space interaction occurred
                    fRootSpaceInteraction->CalculateInteraction();

                    //need to check if an error handler event or stop signal
                    //was triggered during the space calculation
                    if (fStopTrackSignal) {
                        fStep->FinalParticle().SetActive(false);
                    }
                    else {
                        // calculate if a space navigation occurred
                        fRootSpaceNavigator->CalculateNavigation();

                        // if both a space interaction and space navigation occurred, differentiate between them based on which occurred first
                        if ((fStep->GetSpaceInteractionFlag() == true) && (fStep->GetSpaceNavigationFlag() == true)) {

                            // if space interaction was first, execute it and clear space navigation data
                            if (fStep->GetSpaceInteractionStep() < fStep->GetSpaceNavigationStep()) {
                                fRootSpaceInteraction->ExecuteInteraction();
                                fStep->SpaceNavigationName().clear();
                                fStep->SpaceNavigationStep() = numeric_limits<double>::max();
                                fStep->SpaceNavigationFlag() = false;

                                // if space interaction killed a particle, the terminator name is the space interaction name
                                if (fStep->FinalParticle().IsActive() == false) {
                                    fStep->TerminatorName() = fStep->SpaceInteractionName();
                                }
                            }
                            // if space navigation was first, execute it and clear space interaction data
                            else {
                                fRootSpaceNavigator->ExecuteNavigation();
                                fStep->SpaceInteractionName().clear();
                                fStep->SpaceInteractionStep() = numeric_limits<double>::max();
                                fStep->SpaceInteractionFlag() = false;

                                // if space navigation killed a particle, the terminator name is the space navigation name
                                if (fStep->FinalParticle().IsActive() == false) {
                                    fStep->TerminatorName() = fStep->SpaceNavigationName();
                                }
                            }
                        }
                        // if only a space interaction occurred, execute it
                        else if (fStep->GetSpaceInteractionFlag() == true) {
                            fRootSpaceInteraction->ExecuteInteraction();

                            // if space interaction killed a particle, the terminator name is the space interaction name
                            if (fStep->FinalParticle().IsActive() == false) {
                                fStep->TerminatorName() = fStep->SpaceInteractionName();
                            }
                        }
                        // if only a space navigation occurred, execute it
                        else if (fStep->GetSpaceNavigationFlag() == true) {
                            fRootSpaceNavigator->ExecuteNavigation();

                            // if space navigation killed a particle, the terminator name is the space navigation name
                            if (fStep->FinalParticle().IsActive() == false) {
                                fStep->TerminatorName() = fStep->SpaceNavigationName();
                            }
                        }
                        // if neither occurred, execute the trajectory
                        else {
                            fRootTrajectory->ExecuteTrajectory();
                        }

                        // execute post-step modification
                        bool hasPostModified = fRootStepModifier->ExecutePostStepModification();
                        if (hasPostModified) {
                            fRootTrajectory->Reset();
                            if (fStep->FinalParticle().IsActive() == false) {
                                fStep->TerminatorName() = fStep->ModifierName();
                                fRestartNavigation = false;  // this avoids losing internal navigation/terminator states
                            }
                        }

                        fStep->EndTiming();

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
            }
            // if the particle is on a surface or side, continue with surface calculations
            else {
                fRootSurfaceInteraction->ExecuteInteraction();

                //need to check if an error handler event or stop signal
                //was triggered during the surface calculation
                if (fStopTrackSignal) {
                    fStep->FinalParticle().SetActive(false);
                }
                else {
                    // if surface interaction killed a particle, the terminator name is the surface interaction name
                    if (fStep->InteractionParticle().IsActive() == false) {
                        fStep->TerminatorName() = fStep->SurfaceInteractionName();
                    }

                    fRootSurfaceNavigator->ExecuteNavigation();

                    // if surface navigation killed a particle, the terminator name is the surface navigation name
                    if (fStep->FinalParticle().IsActive() == false) {
                        fStep->TerminatorName() = fStep->SurfaceNavigationName();
                    }

                    // execute post-step modification
                    bool hasPostModified = fRootStepModifier->ExecutePostStepModification();
                    if (hasPostModified) {
                        fRootTrajectory->Reset();
                        if (fStep->FinalParticle().IsActive() == false) {
                            fStep->TerminatorName() = fStep->ModifierName();
                            fRestartNavigation = false;  // this avoids losing internal navigation/terminator states
                        }
                    }

                    fStep->EndTiming();

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
            }

            fStepIndex++;
        }
        // if the terminators killed the particle, execute them
        else {
            fRootTerminator->ExecuteTermination();

            fStep->FinalParticle().SetActive(false);

            fStep->EndTiming();

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

        // check if particle has turned around
        double tInitialDotProduct =
            fStep->InitialParticle().GetMagneticField().Dot(fStep->InitialParticle().GetMomentum());
        double tFinalDotProduct = fStep->FinalParticle().GetMagneticField().Dot(fStep->FinalParticle().GetMomentum());
        if (tInitialDotProduct * tFinalDotProduct < 0.) {
            fStep->NumberOfTurns() += 1;
        }
    }
    catch (KSUserInterrupt const& e) {
        stepmsg(eInfo) << "Interrupted at step <" << fStep->StepId() << "> (" << e.what() << ")" << eom;
        // terminate current track and stop run
        fStep->FinalParticle().SetActive(false);
        fStep->TerminatorName() = "user_interrupt";
        fStopRunSignal = true;
    }
    catch (KSException const& e) {
        stepmsg(eWarning) << "Failed to execute step <" << fStep->StepId() << "> (" << e.what() << ")" << eom;
        // terminate current track
        fStep->FinalParticle().SetActive(false);
        fStep->TerminatorName() = e.SignalName();
    }
    catch (KGslException const& e) {
        stepmsg(eWarning) << "Failed to execute step <" << fStep->StepId() << "> (" << e.what() << ")" << eom;
        // terminate current track
        fStep->FinalParticle().SetActive(false);
        fStep->TerminatorName() = "gsl_error";
    }
    // NOTE: any other exceptions lead to failure but may be caught higher up (track/event/run)

    // push deupdate in case exceptions prevented it
    if (fStep->State() == eUpdated)
        fStep->PushDeupdate();
    if (fRootSurfaceInteraction->State() == eUpdated)
        fRootSurfaceInteraction->PushDeupdate();
    if (fRootSurfaceNavigator->State() == eUpdated)
        fRootSurfaceNavigator->PushDeupdate();
    if (fRootTerminator->State() == eUpdated)
        fRootTerminator->PushDeupdate();
    if (fRootStepModifier->State() == eUpdated)
        fRootStepModifier->PushDeupdate();

    // label secondaries
    for (auto& tParticleIt : fStep->ParticleQueue()) {
        tParticleIt->SetParentRunId(fRun->RunId());
        tParticleIt->SetParentEventId(fEvent->EventId());
        tParticleIt->SetParentTrackId(fTrack->TrackId());
        tParticleIt->SetParentStepId(fStep->StepId());
    }

    stepmsg_debug("finished step " << fStep->GetStepId() << eom);
    stepmsg_debug("step final particle state: " << eom);
    stepmsg_debug("  final particle space: <"
                  << (fStep->FinalParticle().GetCurrentSpace() ? fStep->FinalParticle().GetCurrentSpace()->GetName()
                                                               : "")
                  << ">" << eom);
    stepmsg_debug("  final particle surface: <"
                  << (fStep->FinalParticle().GetCurrentSurface() ? fStep->FinalParticle().GetCurrentSurface()->GetName()
                                                                 : "")
                  << ">" << eom);
    stepmsg_debug("  final particle side: <"
                  << (fStep->FinalParticle().GetCurrentSide() ? fStep->FinalParticle().GetCurrentSide()->GetName() : "")
                  << ">" << eom);
    stepmsg_debug("  final particle time: <" << fStep->FinalParticle().GetTime() << ">" << eom);
    stepmsg_debug("  final particle length: <" << fStep->FinalParticle().GetLength() << ">" << eom);
    stepmsg_debug("  final particle position: <" << fStep->FinalParticle().GetPosition().X() << ", "
                                                 << fStep->FinalParticle().GetPosition().Y() << ", "
                                                 << fStep->FinalParticle().GetPosition().Z() << ">" << eom);
    stepmsg_debug("  final particle momentum: <" << fStep->FinalParticle().GetMomentum().X() << ", "
                                                 << fStep->FinalParticle().GetMomentum().Y() << ", "
                                                 << fStep->FinalParticle().GetMomentum().Z() << ">" << eom);
    stepmsg_debug("  final particle kinetic energy: <" << fStep->FinalParticle().GetKineticEnergy_eV() << ">" << eom);
    stepmsg_debug("  final particle electric field: <" << fStep->FinalParticle().GetElectricField().X() << ","
                                                       << fStep->FinalParticle().GetElectricField().Y() << ","
                                                       << fStep->FinalParticle().GetElectricField().Z() << ">" << eom);
    stepmsg_debug("  final particle magnetic field: <" << fStep->FinalParticle().GetMagneticField().X() << ","
                                                       << fStep->FinalParticle().GetMagneticField().Y() << ","
                                                       << fStep->FinalParticle().GetMagneticField().Z() << ">" << eom);
    stepmsg_debug("  final particle angle to magnetic field: <" << fStep->FinalParticle().GetPolarAngleToB() << ">"
                                                                << eom);
    stepmsg_debug("  final particle spin: " << fStep->FinalParticle().GetSpin() << eom);
    stepmsg_debug("  final particle spin0: <" << fStep->FinalParticle().GetSpin0() << ">" << eom);
    stepmsg_debug("  final particle aligned spin: <" << fStep->FinalParticle().GetAlignedSpin() << ">" << eom);
    stepmsg_debug("  final particle spin angle: <" << fStep->FinalParticle().GetSpinAngle() << ">" << eom);

    //signal handler terminate particle
    if (fStopRunSignal || fStopEventSignal || fStopTrackSignal) {
        fStep->FinalParticle().SetActive(false);
    }

    //now that the step is completely done, finalize either the space or the surface navigation for the next step
    if (fStep->FinalParticle().IsActive() == true) {
        if (fStep->GetSpaceNavigationFlag() == true) {
            fRootSpaceNavigator->FinalizeNavigation();
        }
        else if (fStep->GetSurfaceNavigationFlag() == true) {
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
    mainmsg(eWarning) << "stop requested by signal <" << aSignal << ">. stopping simulation..." << eom;

    // first signal stops tracking
    // second signal terminates the program immediately.
    signal(SIGINT, SIG_DFL);
    signal(SIGTERM, SIG_DFL);
    signal(SIGQUIT, SIG_DFL);

    // exception is handled in simulation loop
    fStopRunSignal = true;
    //throw KSUserInterrupt() << "User Interrupt: signal " << aSignal;
}

}  // namespace Kassiopeia
