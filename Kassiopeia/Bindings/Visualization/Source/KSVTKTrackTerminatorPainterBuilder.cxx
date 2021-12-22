#include "KSVTKTrackTerminatorPainterBuilder.h"

#include "KVTKWindowBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

STATICINT sKSVTKTrackTerminatorPainterStructure =
    KSVTKTrackTerminatorPainterBuilder::Attribute<std::string>("name") +
    KSVTKTrackTerminatorPainterBuilder::Attribute<std::string>("file") +
    KSVTKTrackTerminatorPainterBuilder::Attribute<std::string>("path") +
    KSVTKTrackTerminatorPainterBuilder::Attribute<std::string>("outfile") +
    KSVTKTrackTerminatorPainterBuilder::Attribute<std::string>("point_object") +
    KSVTKTrackTerminatorPainterBuilder::Attribute<std::string>("point_variable") +
    KSVTKTrackTerminatorPainterBuilder::Attribute<std::string>("terminator_object") +
    KSVTKTrackTerminatorPainterBuilder::Attribute<std::string>("terminator_variable") +
    KSVTKTrackTerminatorPainterBuilder::Attribute<int>("point_size") +
    KSVTKTrackTerminatorPainterBuilder::Attribute<std::string>("add_terminator") +
    KSVTKTrackTerminatorPainterBuilder::Attribute<KThreeVector>("add_color");

STATICINT sKSVTKTrackTerminatorPainterWindow =
    KVTKWindowBuilder::ComplexElement<KSVTKTrackTerminatorPainter>("vtk_track_terminator_painter");

}  // namespace katrin
