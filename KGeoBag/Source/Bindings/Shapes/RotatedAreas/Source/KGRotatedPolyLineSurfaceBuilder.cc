#include "KGRotatedPolyLineSurfaceBuilder.hh"
#include "KGInterfaceBuilder.hh"

namespace katrin
{

    STATICINT sKGRotatedPolyLineSurfaceBuilder =
        KGInterfaceBuilder::ComplexElement< KGRotatedPolyLineSurface >( "rotated_poly_line_surface" );

    STATICINT sKGRotatedPolyLineSurfaceBuilderStructure =
        KGRotatedPolyLineSurfaceBuilder::Attribute< string >( "name" ) +
        KGRotatedPolyLineSurfaceBuilder::Attribute< unsigned int >( "rotated_mesh_count" ) +
        KGRotatedPolyLineSurfaceBuilder::ComplexElement< KGPlanarPolyLine >( "poly_line" );

}
