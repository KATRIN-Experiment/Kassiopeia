#include "KSGenGeneratorSimulationBuilder.h"

#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

template<> KSGenGeneratorSimulationBuilder::~KComplexElement() = default;

STATICINT sKSGenGeneratorSimulationStructure =
    KSGenGeneratorSimulationBuilder::Attribute<std::string>("name") +
    KSGenGeneratorSimulationBuilder::Attribute<std::string>("base") +
    KSGenGeneratorSimulationBuilder::Attribute<std::string>("path") +
    KSGenGeneratorSimulationBuilder::Attribute<std::string>("position_x") +
    KSGenGeneratorSimulationBuilder::Attribute<std::string>("position_y") +
    KSGenGeneratorSimulationBuilder::Attribute<std::string>("position_z") +
    KSGenGeneratorSimulationBuilder::Attribute<std::string>("direction_x") +
    KSGenGeneratorSimulationBuilder::Attribute<std::string>("direction_y") +
    KSGenGeneratorSimulationBuilder::Attribute<std::string>("direction_z") +
    KSGenGeneratorSimulationBuilder::Attribute<std::string>("energy") +
    KSGenGeneratorSimulationBuilder::Attribute<std::string>("time") +
    KSGenGeneratorSimulationBuilder::Attribute<std::string>("terminator") +
    KSGenGeneratorSimulationBuilder::Attribute<std::string>("generator") +
    KSGenGeneratorSimulationBuilder::Attribute<std::string>("track_group") +
    KSGenGeneratorSimulationBuilder::Attribute<std::string>("position_field") +
    KSGenGeneratorSimulationBuilder::Attribute<std::string>("momentum_field") +
    KSGenGeneratorSimulationBuilder::Attribute<std::string>("kinetic_energy_field") +
    KSGenGeneratorSimulationBuilder::Attribute<std::string>("time_field") +
    KSGenGeneratorSimulationBuilder::Attribute<std::string>("pid_field");

STATICINT sKSGenGeneratorSimulation =
    KSRootBuilder::ComplexElement<KSGenGeneratorSimulation>("ksgen_generator_simulation");

}  // namespace katrin
