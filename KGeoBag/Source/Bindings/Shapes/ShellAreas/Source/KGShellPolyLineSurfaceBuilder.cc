#include "KGShellPolyLineSurfaceBuilder.hh"
#include "KGInterfaceBuilder.hh"

using namespace std;
using namespace KGeoBag;

namespace katrin
{

    STATICINT sKGShellPolyLineSurfaceBuilder =
        KGInterfaceBuilder::ComplexElement< KGShellPolyLineSurface >( "shell_poly_line_surface" );

    STATICINT sKGShellPolyLineSurfaceBuilderStructure =
        KGShellPolyLineSurfaceBuilder::Attribute< string >( "name" ) +
        KGShellPolyLineSurfaceBuilder::Attribute< double >( "angle_start" ) +
        KGShellPolyLineSurfaceBuilder::Attribute< double >( "angle_stop" ) +
        KGShellPolyLineSurfaceBuilder::Attribute< unsigned int >( "shell_mesh_count" ) +
        KGShellPolyLineSurfaceBuilder::Attribute< double >( "shell_mesh_power" ) +
        KGShellPolyLineSurfaceBuilder::ComplexElement< KGPlanarPolyLine >( "poly_line" );

}
