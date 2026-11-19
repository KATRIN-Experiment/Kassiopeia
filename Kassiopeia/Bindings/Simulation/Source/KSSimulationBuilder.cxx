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
    //! the name of the magnetic field used at the start of the simulation.
    //! @reference KMagneticField
    KSSimulationBuilder::Attribute<std::string>("magnetic_field") +
    //! the name of the electric field used at the start of the simulation.
    //! @reference KElectricField
    KSSimulationBuilder::Attribute<std::string>("electric_field") +
    //! the name of the navigation space in which the simulation starts.
    //! @reference KSSpace
    KSSimulationBuilder::Attribute<std::string>("space") +
    //! the name of a navigation surface active at the start of the simulation.
    //! @reference KSSurface
    KSSimulationBuilder::Attribute<std::string>("surface") +
    //! the name of the generator used at the start of the simulation.
    //! @reference KSGenerator
    KSSimulationBuilder::Attribute<std::string>("generator") +
    //! the name of the trajectory used at the start of the simulation.
    //! @reference KSTrajectory
    KSSimulationBuilder::Attribute<std::string>("trajectory") +
    //! the name of a space interaction active at the start of the simulation.
    //! @reference KSSpaceInteraction
    KSSimulationBuilder::Attribute<std::string>("space_interaction") +
    //! the name of the space navigator.
    //! @reference KSSpaceNavigator
    KSSimulationBuilder::Attribute<std::string>("space_navigator") +
    //! the name of a surface interaction active at the start of the simulation.
    //! @reference KSSurfaceInteraction
    KSSimulationBuilder::Attribute<std::string>("surface_interaction") +
    //! the name of the surface navigator.
    //! @reference KSSurfaceNavigator
    KSSimulationBuilder::Attribute<std::string>("surface_navigator") +
    //! the name of a terminator active at the start of the simulation.
    //! @reference KSTerminator
    KSSimulationBuilder::Attribute<std::string>("terminator") +
    //! the name of the writer.
    //! @reference KSWriter
    KSSimulationBuilder::Attribute<std::string>("writer") +
    //! the name of a command that is executed at the start of the simulation.
    KSSimulationBuilder::Attribute<std::string>("command");

}  // namespace katrin
