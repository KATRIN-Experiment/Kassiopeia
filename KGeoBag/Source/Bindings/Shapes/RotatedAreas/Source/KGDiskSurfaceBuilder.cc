#include "KGDiskSurfaceBuilder.hh"
#include "KGInterfaceBuilder.hh"

namespace katrin
{

    static int sKGDiskSurfaceBuilder =
        KGInterfaceBuilder::ComplexElement< KGDiskSurface >( "disk_surface" );

    static int sKGDiskSurfaceBuilderStructure =
        KGDiskSurfaceBuilder::Attribute< string >( "name" ) +
        KGDiskSurfaceBuilder::Attribute< double >( "r" ) +
        KGDiskSurfaceBuilder::Attribute< double >( "z" ) +
        KGDiskSurfaceBuilder::Attribute< unsigned int >( "radial_mesh_count" ) +
        KGDiskSurfaceBuilder::Attribute< double >( "radial_mesh_power" ) +
        KGDiskSurfaceBuilder::Attribute< unsigned int >( "axial_mesh_count" );

}
