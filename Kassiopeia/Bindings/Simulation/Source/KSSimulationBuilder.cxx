#include "KSSimulationBuilder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
namespace katrin
{

    template< >
    KSSimulationBuilder::~KComplexElement()
    {
    }

    static int sKSSimulation =
        KSRootBuilder::ComplexElement< KSSimulation >( "ks_simulation" );

    static int sKSSimulationStructure =
        KSSimulationBuilder::Attribute< string >( "name" ) +
        KSSimulationBuilder::Attribute< unsigned int >( "seed" ) +
        KSSimulationBuilder::Attribute< unsigned int >( "run" ) +
        KSSimulationBuilder::Attribute< unsigned int >( "events" ) +
        KSSimulationBuilder::Attribute< string >( "magnetic_field" ) +
        KSSimulationBuilder::Attribute< string >( "electric_field" ) +
        KSSimulationBuilder::Attribute< string >( "space" ) +
        KSSimulationBuilder::Attribute< string >( "surface" ) +
        KSSimulationBuilder::Attribute< string >( "generator" ) +
        KSSimulationBuilder::Attribute< string >( "trajectory" ) +
        KSSimulationBuilder::Attribute< string >( "space_interaction" ) +
        KSSimulationBuilder::Attribute< string >( "space_navigator" ) +
        KSSimulationBuilder::Attribute< string >( "surface_interaction" ) +
        KSSimulationBuilder::Attribute< string >( "surface_navigator" ) +
        KSSimulationBuilder::Attribute< string >( "terminator" ) +
        KSSimulationBuilder::Attribute< string >( "writer" )+
        KSSimulationBuilder::Attribute< string >( "command" );

}
