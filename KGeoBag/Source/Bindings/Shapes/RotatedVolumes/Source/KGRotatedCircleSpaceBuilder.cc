#include "KGRotatedCircleSpaceBuilder.hh"
#include "KGInterfaceBuilder.hh"

namespace katrin
{

    static const int sKGRotatedCircleSpaceBuilder =
        KGInterfaceBuilder::ComplexElement< KGRotatedCircleSpace >( "rotated_circle_space" );

    static const int sKGRotatedCircleSpaceBuilderStructure =
        KGRotatedCircleSpaceBuilder::Attribute< string >( "name" ) +
        KGRotatedCircleSpaceBuilder::Attribute< unsigned int >( "rotated_mesh_count" ) +
        KGRotatedCircleSpaceBuilder::ComplexElement< KGPlanarCircle >( "circle" );

}
