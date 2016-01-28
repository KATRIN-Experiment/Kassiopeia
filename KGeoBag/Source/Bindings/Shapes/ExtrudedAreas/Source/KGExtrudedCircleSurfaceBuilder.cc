#include "KGExtrudedCircleSurfaceBuilder.hh"
#include "KGInterfaceBuilder.hh"

namespace katrin
{

    STATICINT sKGExtrudedCircleSurfaceBuilder =
        KGInterfaceBuilder::ComplexElement< KGExtrudedCircleSurface >( "extruded_circle_surface" );

    STATICINT sKGExtrudedCircleSurfaceBuilderStructure =
        KGExtrudedCircleSurfaceBuilder::Attribute< string >( "name" ) +
        KGExtrudedCircleSurfaceBuilder::Attribute< double >( "zmin" ) +
        KGExtrudedCircleSurfaceBuilder::Attribute< double >( "zmax" ) +
        KGExtrudedCircleSurfaceBuilder::Attribute< unsigned int >( "extruded_mesh_count" ) +
        KGExtrudedCircleSurfaceBuilder::Attribute< double >( "extruded_mesh_power" ) +
        KGExtrudedCircleSurfaceBuilder::ComplexElement< KGPlanarCircle >( "circle" );

}
