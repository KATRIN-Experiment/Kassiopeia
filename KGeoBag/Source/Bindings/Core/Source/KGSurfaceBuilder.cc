#include "KGSurfaceBuilder.hh"
#include "KGTransformationBuilder.hh"
#include "KGInterfaceBuilder.hh"

using namespace std;
using namespace KGeoBag;

namespace katrin
{
    STATICINT sKGSurfaceBuilder =
        KGInterfaceBuilder::ComplexElement< KGSurface >( "surface" );

    STATICINT sKGSurfaceBuilderStructure =
        KGSurfaceBuilder::Attribute< string >( "name" ) +
        KGSurfaceBuilder::Attribute< string >( "node" ) +
        KGSurfaceBuilder::ComplexElement< KTransformation >( "transformation" );
}
