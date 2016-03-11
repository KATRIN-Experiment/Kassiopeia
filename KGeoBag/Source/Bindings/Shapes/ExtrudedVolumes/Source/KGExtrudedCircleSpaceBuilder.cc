#include "KGExtrudedCircleSpaceBuilder.hh"
#include "KGInterfaceBuilder.hh"

namespace katrin
{

    STATICINT sKGExtrudedCircleSpaceBuilder =
        KGInterfaceBuilder::ComplexElement< KGExtrudedCircleSpace >( "extruded_circle_space" );

    STATICINT sKGExtrudedCircleSpaceBuilderStructure =
        KGExtrudedCircleSpaceBuilder::Attribute< string >( "name" ) +
        KGExtrudedCircleSpaceBuilder::Attribute< double >( "zmin" ) +
        KGExtrudedCircleSpaceBuilder::Attribute< double >( "zmax" ) +
        KGExtrudedCircleSpaceBuilder::Attribute< unsigned int >( "extruded_mesh_count" ) +
        KGExtrudedCircleSpaceBuilder::Attribute< double >( "extruded_mesh_power" ) +
        KGExtrudedCircleSpaceBuilder::Attribute< unsigned int >( "flattened_mesh_count" ) +
        KGExtrudedCircleSpaceBuilder::Attribute< double >( "flattened_mesh_power" ) +
        KGExtrudedCircleSpaceBuilder::ComplexElement< KGPlanarCircle >( "circle" );

}
