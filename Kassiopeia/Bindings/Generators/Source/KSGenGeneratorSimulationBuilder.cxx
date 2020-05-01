#include "KSGenGeneratorSimulationBuilder.h"

#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

template<> KSGenGeneratorSimulationBuilder::~KComplexElement() {}

STATICINT sKSGenGeneratorSimulationStructure =
    KSGenGeneratorSimulationBuilder::Attribute<string>("name") +
    KSGenGeneratorSimulationBuilder::Attribute<string>("base") +
    KSGenGeneratorSimulationBuilder::Attribute<string>("path") +
    KSGenGeneratorSimulationBuilder::Attribute<string>("position_x") +
    KSGenGeneratorSimulationBuilder::Attribute<string>("position_y") +
    KSGenGeneratorSimulationBuilder::Attribute<string>("position_z") +
    KSGenGeneratorSimulationBuilder::Attribute<string>("direction_x") +
    KSGenGeneratorSimulationBuilder::Attribute<string>("direction_y") +
    KSGenGeneratorSimulationBuilder::Attribute<string>("direction_z") +
    KSGenGeneratorSimulationBuilder::Attribute<string>("energy") +
    KSGenGeneratorSimulationBuilder::Attribute<string>("time") +
    KSGenGeneratorSimulationBuilder::Attribute<string>("terminator") +
    KSGenGeneratorSimulationBuilder::Attribute<string>("generator") +
    KSGenGeneratorSimulationBuilder::Attribute<string>("track_group") +
    KSGenGeneratorSimulationBuilder::Attribute<string>("position_field") +
    KSGenGeneratorSimulationBuilder::Attribute<string>("momentum_field") +
    KSGenGeneratorSimulationBuilder::Attribute<string>("kinetic_energy_field") +
    KSGenGeneratorSimulationBuilder::Attribute<string>("time_field") +
    KSGenGeneratorSimulationBuilder::Attribute<string>("pid_field");

STATICINT sKSGenGeneratorSimulation =
    KSRootBuilder::ComplexElement<KSGenGeneratorSimulation>("ksgen_generator_simulation");

}  // namespace katrin
