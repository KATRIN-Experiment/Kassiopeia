#include "KGFlattenedCircleSurfaceBuilder.hh"
#include "KGInterfaceBuilder.hh"

namespace katrin
{

    static const int sKGFlattenedCircleSurfaceBuilder =
        KGInterfaceBuilder::ComplexElement< KGFlattenedCircleSurface >( "flattened_circle_surface" );

    static const int sKGFlattenedCircleSurfaceBuilderStructure =
        KGFlattenedCircleSurfaceBuilder::Attribute< string >( "name" ) +
        KGFlattenedCircleSurfaceBuilder::Attribute< double >( "z" ) +
        KGFlattenedCircleSurfaceBuilder::Attribute< unsigned int >( "flattened_mesh_count" ) +
        KGFlattenedCircleSurfaceBuilder::Attribute< double >( "flattened_mesh_power" ) +
        KGFlattenedCircleSurfaceBuilder::ComplexElement< KGPlanarCircle >( "circle" );

}
