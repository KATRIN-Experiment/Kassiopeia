#include "KGRotatedCircleSurfaceBuilder.hh"
#include "KGInterfaceBuilder.hh"

namespace katrin
{

    static const int sKGRotatedCircleSurfaceBuilder =
        KGInterfaceBuilder::ComplexElement< KGRotatedCircleSurface >( "rotated_circle_surface" );

    static const int sKGRotatedCircleSurfaceBuilderStructure =
        KGRotatedCircleSurfaceBuilder::Attribute< string >( "name" ) +
        KGRotatedCircleSurfaceBuilder::Attribute< unsigned int >( "rotated_mesh_count" ) +
        KGRotatedCircleSurfaceBuilder::ComplexElement< KGPlanarCircle >( "circle" );

}
