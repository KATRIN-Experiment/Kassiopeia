#ifndef Kassiopeia_KSGenGeneratorSimulationBuilder_h_
#define Kassiopeia_KSGenGeneratorSimulationBuilder_h_

#include "KComplexElement.hh"
#include "KSGenGeneratorSimulation.h"
#include "KToolbox.h"

using namespace Kassiopeia;
namespace katrin
{

typedef KComplexElement<KSGenGeneratorSimulation> KSGenGeneratorSimulationBuilder;

template<> inline bool KSGenGeneratorSimulationBuilder::AddAttribute(KContainer* aContainer)
{
    if (aContainer->GetName() == "name") {
        aContainer->CopyTo(fObject, &KNamed::SetName);
        return true;
    }
    // path to the existing simulation output file (e.g. base="DipoleTrapSimulation.root")
    if (aContainer->GetName() == "base") {
        aContainer->CopyTo(fObject, &KSGenGeneratorSimulation::SetBase);
        return true;
    }
    if (aContainer->GetName() == "path") {
        aContainer->CopyTo(fObject, &KSGenGeneratorSimulation::SetPath);
        return true;
    }
    // formulas to adjust the generated values (e.g. energy="x + 18600")
    if (aContainer->GetName() == "position_x") {
        aContainer->CopyTo(fObject, &KSGenGeneratorSimulation::SetPositionX);
        return true;
    }
    if (aContainer->GetName() == "position_y") {
        aContainer->CopyTo(fObject, &KSGenGeneratorSimulation::SetPositionY);
        return true;
    }
    if (aContainer->GetName() == "position_z") {
        aContainer->CopyTo(fObject, &KSGenGeneratorSimulation::SetPositionZ);
        return true;
    }
    if (aContainer->GetName() == "direction_x") {
        aContainer->CopyTo(fObject, &KSGenGeneratorSimulation::SetDirectionX);
        return true;
    }
    if (aContainer->GetName() == "direction_y") {
        aContainer->CopyTo(fObject, &KSGenGeneratorSimulation::SetDirectionY);
        return true;
    }
    if (aContainer->GetName() == "direction_z") {
        aContainer->CopyTo(fObject, &KSGenGeneratorSimulation::SetDirectionZ);
        return true;
    }
    if (aContainer->GetName() == "energy") {
        aContainer->CopyTo(fObject, &KSGenGeneratorSimulation::SetEnergy);
        return true;
    }
    if (aContainer->GetName() == "time") {
        aContainer->CopyTo(fObject, &KSGenGeneratorSimulation::SetTime);
        return true;
    }
    // names of the generator/terminator to select tracks (e.g. terminator="term_detector")
    if (aContainer->GetName() == "terminator") {
        aContainer->CopyTo(fObject, &KSGenGeneratorSimulation::SetTerminator);
        return true;
    }
    if (aContainer->GetName() == "generator") {
        aContainer->CopyTo(fObject, &KSGenGeneratorSimulation::SetGenerator);
        return true;
    }
    // name of the track group to use (e.g. track_group="output_track_world")
    if (aContainer->GetName() == "track_group") {
        aContainer->CopyTo(fObject, &KSGenGeneratorSimulation::SetTrackGroupName);
        return true;
    }
    // names of the track components to use (e.g. position_field="final_position")
    if (aContainer->GetName() == "position_field") {
        aContainer->CopyTo(fObject, &KSGenGeneratorSimulation::SetPositionName);
        return true;
    }
    if (aContainer->GetName() == "momentum_field") {
        aContainer->CopyTo(fObject, &KSGenGeneratorSimulation::SetMomentumName);
        return true;
    }
    if (aContainer->GetName() == "kinetic_energy_field") {
        aContainer->CopyTo(fObject, &KSGenGeneratorSimulation::SetKineticEnergyName);
        return true;
    }
    if (aContainer->GetName() == "time_field") {
        aContainer->CopyTo(fObject, &KSGenGeneratorSimulation::SetTimeName);
        return true;
    }
    if (aContainer->GetName() == "pid_field") {
        aContainer->CopyTo(fObject, &KSGenGeneratorSimulation::SetPIDName);
        return true;
    }
    return false;
}

}  // namespace katrin
#endif
