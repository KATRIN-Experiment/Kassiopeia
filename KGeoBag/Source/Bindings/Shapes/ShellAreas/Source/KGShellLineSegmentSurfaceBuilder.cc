#include "KGShellLineSegmentSurfaceBuilder.hh"
#include "KGInterfaceBuilder.hh"

using namespace std;
using namespace KGeoBag;

namespace katrin
{

    STATICINT sKGShellLineSegmentSurfaceBuilder =
        KGInterfaceBuilder::ComplexElement< KGShellLineSegmentSurface >( "shell_line_segment_surface" );

    STATICINT sKGShellLineSegmentSurfaceBuilderStructure =
        KGShellLineSegmentSurfaceBuilder::Attribute< string >( "name" ) +
        KGShellLineSegmentSurfaceBuilder::Attribute< double >( "angle_start" ) +
        KGShellLineSegmentSurfaceBuilder::Attribute< double >( "angle_stop" ) +
        KGShellLineSegmentSurfaceBuilder::Attribute< unsigned int >( "shell_mesh_count" ) +
        KGShellLineSegmentSurfaceBuilder::Attribute< double >( "shell_mesh_power" ) +
        KGShellLineSegmentSurfaceBuilder::ComplexElement< KGPlanarLineSegment >( "line_segment" );

}
