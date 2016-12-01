#include "KGSpaceBuilder.hh"
#include "KGSurfaceBuilder.hh"
#include "KGInterfaceBuilder.hh"
#include "KGTransformationBuilder.hh"

using namespace std;
using namespace KGeoBag;

namespace katrin
{
    STATICINT sKGSpaceBuilder =
        KGInterfaceBuilder::ComplexElement< KGSpace >( "space" );

    STATICINT sKGSpaceBuilderStructure =
        KGSpaceBuilder::Attribute< string >( "name" ) +
        KGSpaceBuilder::Attribute< string >( "node" ) +
        KGSpaceBuilder::Attribute< string >( "tree" ) +
        KGSpaceBuilder::ComplexElement< KTransformation >( "transformation" ) +
        KGSpaceBuilder::ComplexElement< KGSpace >( "space" ) +
        KGSpaceBuilder::ComplexElement< KGSurface >( "surface" );
}
