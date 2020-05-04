#include "KSVTKTrackTerminatorPainterBuilder.h"

#include "KVTKWindowBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

STATICINT sKSVTKTrackTerminatorPainterStructure =
    KSVTKTrackTerminatorPainterBuilder::Attribute<string>("name") +
    KSVTKTrackTerminatorPainterBuilder::Attribute<string>("file") +
    KSVTKTrackTerminatorPainterBuilder::Attribute<string>("path") +
    KSVTKTrackTerminatorPainterBuilder::Attribute<string>("outfile") +
    KSVTKTrackTerminatorPainterBuilder::Attribute<string>("point_object") +
    KSVTKTrackTerminatorPainterBuilder::Attribute<string>("point_variable") +
    KSVTKTrackTerminatorPainterBuilder::Attribute<string>("terminator_object") +
    KSVTKTrackTerminatorPainterBuilder::Attribute<string>("terminator_variable") +
    KSVTKTrackTerminatorPainterBuilder::Attribute<int>("point_size") +
    KSVTKTrackTerminatorPainterBuilder::Attribute<string>("add_terminator") +
    KSVTKTrackTerminatorPainterBuilder::Attribute<KThreeVector>("add_color");

STATICINT sKSVTKTrackTerminatorPainterWindow =
    KVTKWindowBuilder::ComplexElement<KSVTKTrackTerminatorPainter>("vtk_track_terminator_painter");

}  // namespace katrin
