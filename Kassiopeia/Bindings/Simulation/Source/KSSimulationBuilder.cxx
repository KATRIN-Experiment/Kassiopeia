#include "KSSimulationBuilder.h"

#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

template<> KSSimulationBuilder::~KComplexElement() = default;

STATICINT sKSSimulation = KSRootBuilder::ComplexElement<KSSimulation>("ks_simulation");

STATICINT sKSSimulationStructure =
    KSSimulationBuilder::Attribute<std::string>("name") + KSSimulationBuilder::Attribute<unsigned int>("seed") +
    KSSimulationBuilder::Attribute<unsigned int>("run") + KSSimulationBuilder::Attribute<unsigned int>("events") +
    KSSimulationBuilder::Attribute<unsigned int>("step_report_iteration") +
    KSSimulationBuilder::Attribute<std::string>("add_static_run_modifier") +
    KSSimulationBuilder::Attribute<std::string>("add_static_event_modifier") +
    KSSimulationBuilder::Attribute<std::string>("add_static_track_modifier") +
    KSSimulationBuilder::Attribute<std::string>("add_static_step_modifier") +
    KSSimulationBuilder::Attribute<std::string>("magnetic_field") +
    KSSimulationBuilder::Attribute<std::string>("electric_field") +
    KSSimulationBuilder::Attribute<std::string>("space") + KSSimulationBuilder::Attribute<std::string>("surface") +
    KSSimulationBuilder::Attribute<std::string>("generator") +
    KSSimulationBuilder::Attribute<std::string>("trajectory") +
    KSSimulationBuilder::Attribute<std::string>("space_interaction") +
    KSSimulationBuilder::Attribute<std::string>("space_navigator") +
    KSSimulationBuilder::Attribute<std::string>("surface_interaction") +
    KSSimulationBuilder::Attribute<std::string>("surface_navigator") +
    KSSimulationBuilder::Attribute<std::string>("terminator") + KSSimulationBuilder::Attribute<std::string>("writer") +
    KSSimulationBuilder::Attribute<std::string>("command");

}  // namespace katrin
