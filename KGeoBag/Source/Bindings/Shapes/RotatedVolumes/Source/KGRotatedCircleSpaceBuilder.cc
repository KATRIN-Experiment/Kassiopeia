#include "KGRotatedCircleSpaceBuilder.hh"
#include "KGInterfaceBuilder.hh"

namespace katrin
{

    STATICINT sKGRotatedCircleSpaceBuilder =
        KGInterfaceBuilder::ComplexElement< KGRotatedCircleSpace >( "rotated_circle_space" );

    STATICINT sKGRotatedCircleSpaceBuilderStructure =
        KGRotatedCircleSpaceBuilder::Attribute< string >( "name" ) +
        KGRotatedCircleSpaceBuilder::Attribute< unsigned int >( "rotated_mesh_count" ) +
        KGRotatedCircleSpaceBuilder::ComplexElement< KGPlanarCircle >( "circle" );

}
