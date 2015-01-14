#include "KGRotatedPolyLineSpaceBuilder.hh"
#include "KGInterfaceBuilder.hh"

namespace katrin
{

    static const int sKGRotatedPolyLineSpaceBuilder =
        KGInterfaceBuilder::ComplexElement< KGRotatedPolyLineSpace >( "rotated_poly_line_space" );

    static const int sKGRotatedPolyLineSpaceBuilderStructure =
        KGRotatedPolyLineSpaceBuilder::Attribute< string >( "name" ) +
        KGRotatedPolyLineSpaceBuilder::Attribute< unsigned int >( "rotated_mesh_count" ) +
        KGRotatedPolyLineSpaceBuilder::Attribute< unsigned int >( "flattened_mesh_count" ) +
        KGRotatedPolyLineSpaceBuilder::Attribute< double >( "flattened_mesh_power" ) +
        KGRotatedPolyLineSpaceBuilder::ComplexElement< KGPlanarPolyLine >( "poly_line" );

}
