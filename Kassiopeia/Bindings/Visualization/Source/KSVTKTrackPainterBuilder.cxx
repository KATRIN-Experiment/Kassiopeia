#include "KSVTKTrackPainterBuilder.h"

#include "KVTKWindowBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

STATICINT sKSVTKTrackPainterStructure = KSVTKTrackPainterBuilder::Attribute<std::string>("name") +
                                        KSVTKTrackPainterBuilder::Attribute<std::string>("file") +
                                        KSVTKTrackPainterBuilder::Attribute<std::string>("path") +
                                        KSVTKTrackPainterBuilder::Attribute<std::string>("outfile") +
                                        KSVTKTrackPainterBuilder::Attribute<std::string>("point_object") +
                                        KSVTKTrackPainterBuilder::Attribute<std::string>("point_variable") +
                                        KSVTKTrackPainterBuilder::Attribute<std::string>("color_object") +
                                        KSVTKTrackPainterBuilder::Attribute<std::string>("color_variable") +
                                        KSVTKTrackPainterBuilder::Attribute<int>("line_width");

STATICINT sKSVTKTrackPainterWindow = KVTKWindowBuilder::ComplexElement<KSVTKTrackPainter>("vtk_track_painter");

}  // namespace katrin
