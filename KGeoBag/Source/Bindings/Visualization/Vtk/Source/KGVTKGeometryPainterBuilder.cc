#include "KGVTKGeometryPainterBuilder.hh"

#include "KVTKWindowBuilder.h"

using namespace KGeoBag;
using namespace std;

namespace katrin
{

STATICINT sKGVTKGeometryPainterStructure = KGVTKGeometryPainterBuilder::Attribute<std::string>("name") +
                                           KGVTKGeometryPainterBuilder::Attribute<std::string>("file") +
                                           KGVTKGeometryPainterBuilder::Attribute<std::string>("path") +
                                           KGVTKGeometryPainterBuilder::Attribute<bool>("write_stl") +
                                           KGVTKGeometryPainterBuilder::Attribute<std::string>("surfaces") +
                                           KGVTKGeometryPainterBuilder::Attribute<std::string>("spaces");

STATICINT sKGVTKGeometryPainterWindow = KVTKWindowBuilder::ComplexElement<KGVTKGeometryPainter>("vtk_geometry_painter");

}  // namespace katrin
