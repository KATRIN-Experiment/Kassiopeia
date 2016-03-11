#include "KSVTKTrackTerminatorPainterBuilder.h"
#include "KVTKWindow.h"

using namespace Kassiopeia;
namespace katrin
{

    STATICINT sKSVTKTrackTerminatorPainterStructure =
        KSVTKTrackTerminatorPainterBuilder::Attribute< string >( "name" ) +
        KSVTKTrackTerminatorPainterBuilder::Attribute< string >( "file" ) +
        KSVTKTrackTerminatorPainterBuilder::Attribute< string >( "path" ) +
        KSVTKTrackTerminatorPainterBuilder::Attribute< string >( "outfile" ) +
        KSVTKTrackTerminatorPainterBuilder::Attribute< string >( "point_object" ) +
        KSVTKTrackTerminatorPainterBuilder::Attribute< string >( "point_variable" ) +
        KSVTKTrackTerminatorPainterBuilder::Attribute< string >( "terminator_object" ) +
        KSVTKTrackTerminatorPainterBuilder::Attribute< string >( "terminator_variable" ) +
        KSVTKTrackTerminatorPainterBuilder::Attribute< int >( "point_size" ) +
        KSVTKTrackTerminatorPainterBuilder::Attribute< string >( "add_terminator" ) +
        KSVTKTrackTerminatorPainterBuilder::Attribute< KThreeVector >( "add_color" );

    STATICINT sKSVTKTrackTerminatorPainterWindow =
        KVTKWindowBuilder::ComplexElement< KSVTKTrackTerminatorPainter >( "vtk_track_terminator_painter" );

}
