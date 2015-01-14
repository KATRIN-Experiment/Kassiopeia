#include "KGAnnulusSurfaceBuilder.hh"
#include "KGInterfaceBuilder.hh"

namespace katrin
{

    static const int sKGAnnulusSurfaceBuilder =
        KGInterfaceBuilder::ComplexElement< KGAnnulusSurface >( "annulus_surface" );

    static const int sKGAnnulusSurfaceBuilderStructure =
        KGAnnulusSurfaceBuilder::Attribute< string >( "name" ) +
        KGAnnulusSurfaceBuilder::Attribute< double >( "z" ) +
        KGAnnulusSurfaceBuilder::Attribute< double >( "r1" ) +
        KGAnnulusSurfaceBuilder::Attribute< double >( "r2" ) +
        KGAnnulusSurfaceBuilder::Attribute< unsigned int >( "radial_mesh_count" ) +
        KGAnnulusSurfaceBuilder::Attribute< double >( "radial_mesh_power" ) +
        KGAnnulusSurfaceBuilder::Attribute< unsigned int >( "axial_mesh_count" );

}
