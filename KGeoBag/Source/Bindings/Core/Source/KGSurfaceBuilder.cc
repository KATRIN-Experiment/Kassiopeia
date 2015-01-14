#include "KGSurfaceBuilder.hh"
#include "KGTransformationBuilder.hh"
#include "KGInterfaceBuilder.hh"

namespace katrin
{
    static int sKGSurfaceBuilder =
        KGInterfaceBuilder::ComplexElement< KGSurface >( "surface" );

    static int sKGSurfaceBuilderStructure =
        KGSurfaceBuilder::Attribute< string >( "name" ) +
        KGSurfaceBuilder::Attribute< string >( "node" ) +
        KGSurfaceBuilder::ComplexElement< KTransformation >( "transformation" );
}
