#include "KGExtrudedPolyLineSurfaceBuilder.hh"
#include "KGInterfaceBuilder.hh"

namespace katrin
{

    static const int sKGExtrudedPolyLineSurfaceBuilder =
        KGInterfaceBuilder::ComplexElement< KGExtrudedPolyLineSurface >( "extruded_poly_line_surface" );

    static const int sKGExtrudedPolyLineSurfaceBuilderStructure =
        KGExtrudedPolyLineSurfaceBuilder::Attribute< string >( "name" ) +
        KGExtrudedPolyLineSurfaceBuilder::Attribute< double >( "zmin" ) +
        KGExtrudedPolyLineSurfaceBuilder::Attribute< double >( "zmax" ) +
        KGExtrudedPolyLineSurfaceBuilder::Attribute< unsigned int >( "extruded_mesh_count" ) +
        KGExtrudedPolyLineSurfaceBuilder::Attribute< double >( "extruded_mesh_power" ) +
        KGExtrudedPolyLineSurfaceBuilder::ComplexElement< KGPlanarPolyLine >( "poly_line" );

}
