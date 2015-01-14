#include "KGRotatedPolyLoopSpaceBuilder.hh"
#include "KGInterfaceBuilder.hh"

namespace katrin
{

    static const int sKGRotatedPolyLoopSpaceBuilder =
        KGInterfaceBuilder::ComplexElement< KGRotatedPolyLoopSpace >( "rotated_poly_loop_space" );

    static const int sKGRotatedPolyLoopSpaceBuilderStructure =
        KGRotatedPolyLoopSpaceBuilder::Attribute< string >( "name" ) +
        KGRotatedPolyLoopSpaceBuilder::Attribute< unsigned int >( "rotated_mesh_count" ) +
        KGRotatedPolyLoopSpaceBuilder::ComplexElement< KGPlanarPolyLoop >( "poly_loop" );

}
