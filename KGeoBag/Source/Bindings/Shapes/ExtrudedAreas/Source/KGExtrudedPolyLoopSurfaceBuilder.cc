#include "KGExtrudedPolyLoopSurfaceBuilder.hh"
#include "KGInterfaceBuilder.hh"

using namespace std;
using namespace KGeoBag;

namespace katrin
{

    STATICINT sKGExtrudedPolyLoopSurfaceBuilder =
        KGInterfaceBuilder::ComplexElement< KGExtrudedPolyLoopSurface >( "extruded_poly_loop_surface" );

    STATICINT sKGExtrudedPolyLoopSurfaceBuilderStructure =
        KGExtrudedPolyLoopSurfaceBuilder::Attribute< string >( "name" ) +
        KGExtrudedPolyLoopSurfaceBuilder::Attribute< double >( "zmin" ) +
        KGExtrudedPolyLoopSurfaceBuilder::Attribute< double >( "zmax" ) +
        KGExtrudedPolyLoopSurfaceBuilder::Attribute< unsigned int >( "extruded_mesh_count" ) +
        KGExtrudedPolyLoopSurfaceBuilder::Attribute< double >( "extruded_mesh_power" ) +
        KGExtrudedPolyLoopSurfaceBuilder::ComplexElement< KGPlanarPolyLoop >( "poly_loop" );

}
