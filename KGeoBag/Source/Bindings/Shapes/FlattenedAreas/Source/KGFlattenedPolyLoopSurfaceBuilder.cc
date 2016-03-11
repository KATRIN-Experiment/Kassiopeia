#include "KGFlattenedPolyLoopSurfaceBuilder.hh"
#include "KGInterfaceBuilder.hh"

namespace katrin
{

    STATICINT sKGFlattenedPolyLoopSurfaceBuilder =
        KGInterfaceBuilder::ComplexElement< KGFlattenedPolyLoopSurface >( "flattened_poly_loop_surface" );

    STATICINT sKGFlattenedPolyLoopSurfaceBuilderStructure =
        KGFlattenedPolyLoopSurfaceBuilder::Attribute< string >( "name" ) +
        KGFlattenedPolyLoopSurfaceBuilder::Attribute< double >( "z" ) +
        KGFlattenedPolyLoopSurfaceBuilder::Attribute< unsigned int >( "flattened_mesh_count" ) +
        KGFlattenedPolyLoopSurfaceBuilder::Attribute< double >( "flattened_mesh_power" ) +
        KGFlattenedPolyLoopSurfaceBuilder::ComplexElement< KGPlanarPolyLoop >( "poly_loop" );

}
