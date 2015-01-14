#include "KGFlattenedPolyLoopSurfaceBuilder.hh"
#include "KGInterfaceBuilder.hh"

namespace katrin
{

    static const int sKGFlattenedPolyLoopSurfaceBuilder =
        KGInterfaceBuilder::ComplexElement< KGFlattenedPolyLoopSurface >( "flattened_poly_loop_surface" );

    static const int sKGFlattenedPolyLoopSurfaceBuilderStructure =
        KGFlattenedPolyLoopSurfaceBuilder::Attribute< string >( "name" ) +
        KGFlattenedPolyLoopSurfaceBuilder::Attribute< double >( "z" ) +
        KGFlattenedPolyLoopSurfaceBuilder::Attribute< unsigned int >( "flattened_mesh_count" ) +
        KGFlattenedPolyLoopSurfaceBuilder::Attribute< double >( "flattened_mesh_power" ) +
        KGFlattenedPolyLoopSurfaceBuilder::ComplexElement< KGPlanarPolyLoop >( "poly_loop" );

}
