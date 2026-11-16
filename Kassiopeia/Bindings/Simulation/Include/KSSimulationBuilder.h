#ifndef Kassiopeia_KSSimulationBuilder_h_
#define Kassiopeia_KSSimulationBuilder_h_

#include "KComplexElement.hh"
#include "KSEventModifier.h"
#include "KSFieldFinder.h"
#include "KSMainMessage.h"
#include "KSRootElectricField.h"
#include "KSRootGenerator.h"
#include "KSRootMagneticField.h"
#include "KSRootSpace.h"
#include "KSRootSpaceInteraction.h"
#include "KSRootSpaceNavigator.h"
#include "KSRootSurfaceInteraction.h"
#include "KSRootSurfaceNavigator.h"
#include "KSRootTerminator.h"
#include "KSRootTrajectory.h"
#include "KSRootWriter.h"
#include "KSRunModifier.h"
#include "KSSimulation.h"
#include "KSStepModifier.h"
#include "KSTrackModifier.h"
#include "KToolbox.h"

using namespace Kassiopeia;

namespace katrin
{

//! The final object to be declared within ``<kassiopeia>`` is the simulation element. This describes the simulation object
//! **KSSimulation**, which is then executed by **KSRoot**. The simulation element specifies the global
//! and initial properties of the simulation as a whole. For example in the ``QuadrupoleTrapSimulation.xml`` example, the
//! simulation element is declared as follows:
//!
//! .. code-block:: xml
//!
//!     <ks_simulation
//!         name="quadrupole_trap_simulation"
//!         run="1"
//!         seed="51385"
//!         events="10"
//!         magnetic_field="field_electromagnet"
//!         electric_field="field_electrostatic"
//!         space="space_world"
//!         generator="generator_uniform"
//!         trajectory="trajectory_exact"
//!         space_navigator="nav_space"
//!         surface_navigator="nav_surface"
//!         writer="write_root"
//!     />
//!
//! The ``run`` is simply a user provided identifier. The ``seed`` is the value provided to the global (singleton) random
//! number generator. Simulations with the same configuration and same seed should provide identical results. If the user is
//! interested in running *Kassiopeia* on many machines in order to achieve high throughput particle tracking (Monte Carlo),
//! care must be taken to ensure that the ``seed`` value is different for each run of the simulation.
//!
//! The parameter ``events`` determines the total number of times that the generator is run (however this is not necessarily
//! the number of particles that will be tracked, e.g. if lists or sets are used in the generator of if secondary particles
//! are created). The remaining parameters ``magnetic_field``, ``space``, ``generator``, etc. all specify the default
//! objects to be used for the initial state of the simulation; where commands specified within ``ksgeo_space`` may
//! modify the actual objects used during the course of a simulation.
//!
//! Following the declaration of ``ks_simulation``, the closing tag ``</kassiopeia>`` is placed to complete the simulation
//! configuration. When this tag is encountered by the XML parser, it triggers the simulation to run.
typedef KComplexElement<KSSimulation> KSSimulationBuilder;

template<> inline bool KSSimulationBuilder::AddAttribute(KContainer* aContainer)
{
    if (aContainer->GetName() == "name") {
        aContainer->CopyTo(fObject, &KSSimulation::SetName);
        return true;
    }
    if (aContainer->GetName() == "seed") {
        aContainer->CopyTo(fObject, &KSSimulation::SetSeed);
        return true;
    }
    if (aContainer->GetName() == "run") {
        aContainer->CopyTo(fObject, &KSSimulation::SetRun);
        return true;
    }
    if (aContainer->GetName() == "events") {
        aContainer->CopyTo(fObject, &KSSimulation::SetEvents);
        return true;
    }
    if (aContainer->GetName() == "step_report_iteration") {
        if (aContainer->AsReference<unsigned int>() == 0) {
            mainmsg(eError) << "Paramter of attribute <"
                            << "step_report_iteration"
                            << "> should not be 0" << eom;
        }
        aContainer->CopyTo(fObject, &KSSimulation::SetStepReportIteration);
        return true;
    }
    if (aContainer->GetName() == "add_static_run_modifier") {
        fObject->AddStaticRunModifier(KToolbox::GetInstance().Get<KSRunModifier>(aContainer->AsString()));
        return true;
    }
    if (aContainer->GetName() == "add_static_event_modifier") {
        fObject->AddStaticEventModifier(KToolbox::GetInstance().Get<KSEventModifier>(aContainer->AsString()));
        return true;
    }
    if (aContainer->GetName() == "add_static_track_modifier") {
        fObject->AddStaticTrackModifier(KToolbox::GetInstance().Get<KSTrackModifier>(aContainer->AsString()));
        return true;
    }
    if (aContainer->GetName() == "add_static_step_modifier") {
        fObject->AddStaticStepModifier(KToolbox::GetInstance().Get<KSStepModifier>(aContainer->AsString()));
        return true;
    }
    if (aContainer->GetName() == "magnetic_field") {
        KSComponent* tComponent = dynamic_cast<KSComponent*>(getMagneticField(aContainer->AsString()));
        KSCommand* tCommand = KToolbox::GetInstance()
                                  .Get<KSRootMagneticField>("root_magnetic_field")
                                  ->Command("add_magnetic_field", tComponent);
        fObject->AddCommand(tCommand);
        return true;
    }
    if (aContainer->GetName() == "electric_field") {
        KSComponent* tComponent = dynamic_cast<KSComponent*>(getElectricField(aContainer->AsString()));
        KSCommand* tCommand = KToolbox::GetInstance()
                                  .Get<KSRootElectricField>("root_electric_field")
                                  ->Command("add_electric_field", tComponent);
        fObject->AddCommand(tCommand);
        return true;
    }
    if (aContainer->GetName() == "space") {
        auto* tComponent = KToolbox::GetInstance().Get<KSComponent>(aContainer->AsString());
        KSCommand* tCommand = KToolbox::GetInstance().Get<KSRootSpace>("root_space")->Command("add_space", tComponent);
        fObject->AddCommand(tCommand);
        return true;
    }
    if (aContainer->GetName() == "surface") {
        auto* tComponent = KToolbox::GetInstance().Get<KSComponent>(aContainer->AsString());
        KSCommand* tCommand =
            KToolbox::GetInstance().Get<KSRootSpace>("root_space")->Command("add_surface", tComponent);
        fObject->AddCommand(tCommand);
        return true;
    }
    if (aContainer->GetName() == "generator") {
        auto* tComponent = KToolbox::GetInstance().Get<KSComponent>(aContainer->AsString());
        KSCommand* tCommand =
            KToolbox::GetInstance().Get<KSRootGenerator>("root_generator")->Command("set_generator", tComponent);
        fObject->AddCommand(tCommand);
        return true;
    }
    if (aContainer->GetName() == "trajectory") {
        auto* tComponent = KToolbox::GetInstance().Get<KSComponent>(aContainer->AsString());
        KSCommand* tCommand =
            KToolbox::GetInstance().Get<KSRootTrajectory>("root_trajectory")->Command("set_trajectory", tComponent);
        fObject->AddCommand(tCommand);
        return true;
    }
    if (aContainer->GetName() == "space_interaction") {
        auto* tComponent = KToolbox::GetInstance().Get<KSComponent>(aContainer->AsString());
        KSCommand* tCommand = KToolbox::GetInstance()
                                  .Get<KSRootSpaceInteraction>("root_space_interaction")
                                  ->Command("add_space_interaction", tComponent);
        fObject->AddCommand(tCommand);
        return true;
    }
    if (aContainer->GetName() == "space_navigator") {
        auto* tComponent = KToolbox::GetInstance().Get<KSComponent>(aContainer->AsString());
        KSCommand* tCommand = KToolbox::GetInstance()
                                  .Get<KSRootSpaceNavigator>("root_space_navigator")
                                  ->Command("set_space_navigator", tComponent);
        fObject->AddCommand(tCommand);
        return true;
    }
    if (aContainer->GetName() == "surface_interaction") {
        auto* tComponent = KToolbox::GetInstance().Get<KSComponent>(aContainer->AsString());
        KSCommand* tCommand = KToolbox::GetInstance()
                                  .Get<KSRootSurfaceInteraction>("root_surface_interaction")
                                  ->Command("set_surface_interaction", tComponent);
        fObject->AddCommand(tCommand);
        return true;
    }
    if (aContainer->GetName() == "surface_navigator") {
        auto* tComponent = KToolbox::GetInstance().Get<KSComponent>(aContainer->AsString());
        KSCommand* tCommand = KToolbox::GetInstance()
                                  .Get<KSRootSurfaceNavigator>("root_surface_navigator")
                                  ->Command("set_surface_navigator", tComponent);
        fObject->AddCommand(tCommand);
        return true;
    }
    if (aContainer->GetName() == "terminator") {
        auto* tComponent = KToolbox::GetInstance().Get<KSComponent>(aContainer->AsString());
        KSCommand* tCommand =
            KToolbox::GetInstance().Get<KSRootTerminator>("root_terminator")->Command("add_terminator", tComponent);
        fObject->AddCommand(tCommand);
        return true;
    }
    if (aContainer->GetName() == "writer") {
        auto* tComponent = KToolbox::GetInstance().Get<KSComponent>(aContainer->AsString());
        KSCommand* tCommand =
            KToolbox::GetInstance().Get<KSRootWriter>("root_writer")->Command("add_writer", tComponent);
        fObject->AddCommand(tCommand);
        return true;
    }
    if (aContainer->GetName() == "command") {
        fObject->AddCommand(KToolbox::GetInstance().Get<KSCommand>(aContainer->AsString()));
        return true;
    }

    return false;
}

}  // namespace katrin

#endif
