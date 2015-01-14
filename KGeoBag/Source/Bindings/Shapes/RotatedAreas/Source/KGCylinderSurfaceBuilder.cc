#include "KGCylinderSurfaceBuilder.hh"
#include "KGInterfaceBuilder.hh"

namespace katrin
{

    static const int sKGCylinderSurfaceBuilder =
        KGInterfaceBuilder::ComplexElement< KGCylinderSurface >( "cylinder_surface" );

    static const int sKGCylinderSurfaceBuilderStructure =
        KGCylinderSurfaceBuilder::Attribute< string >( "name" ) +
        KGCylinderSurfaceBuilder::Attribute< double >( "z1" ) +
        KGCylinderSurfaceBuilder::Attribute< double >( "z2" ) +
        KGCylinderSurfaceBuilder::Attribute< double >( "length" ) +
        KGCylinderSurfaceBuilder::Attribute< double >( "r" ) +
        KGCylinderSurfaceBuilder::Attribute< unsigned int >( "longitudinal_mesh_count" ) +
        KGCylinderSurfaceBuilder::Attribute< double >( "longitudinal_mesh_power" ) +
        KGCylinderSurfaceBuilder::Attribute< unsigned int >( "axial_mesh_count" );

}
