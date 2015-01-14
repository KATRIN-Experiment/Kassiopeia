#include "KGSpaceBuilder.hh"
#include "KGSurfaceBuilder.hh"
#include "KGInterfaceBuilder.hh"
#include "KGTransformationBuilder.hh"

namespace katrin
{
    static int sKGSpaceBuilder =
        KGInterfaceBuilder::ComplexElement< KGSpace >( "space" );

    static int sKGSpaceBuilderStructure =
        KGSpaceBuilder::Attribute< string >( "name" ) +
        KGSpaceBuilder::Attribute< string >( "node" ) +
        KGSpaceBuilder::Attribute< string >( "tree" ) +
        KGSpaceBuilder::ComplexElement< KTransformation >( "transformation" ) +
        KGSpaceBuilder::ComplexElement< KGSpace >( "space" ) +
        KGSpaceBuilder::ComplexElement< KGSurface >( "surface" );
}
