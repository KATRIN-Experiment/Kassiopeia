#include "KGExtrudedCircleSurfaceBuilder.hh"
#include "KGInterfaceBuilder.hh"

namespace katrin
{

    static const int sKGExtrudedCircleSurfaceBuilder =
        KGInterfaceBuilder::ComplexElement< KGExtrudedCircleSurface >( "extruded_circle_surface" );

    static const int sKGExtrudedCircleSurfaceBuilderStructure =
        KGExtrudedCircleSurfaceBuilder::Attribute< string >( "name" ) +
        KGExtrudedCircleSurfaceBuilder::Attribute< double >( "zmin" ) +
        KGExtrudedCircleSurfaceBuilder::Attribute< double >( "zmax" ) +
        KGExtrudedCircleSurfaceBuilder::Attribute< unsigned int >( "extruded_mesh_count" ) +
        KGExtrudedCircleSurfaceBuilder::Attribute< double >( "extruded_mesh_power" ) +
        KGExtrudedCircleSurfaceBuilder::ComplexElement< KGPlanarCircle >( "circle" );

}
