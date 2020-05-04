#include "KGVTKGeometryPainterBuilder.hh"

#include "KVTKWindowBuilder.h"

using namespace KGeoBag;
using namespace std;

namespace katrin
{

STATICINT sKGVTKGeometryPainterStructure = KGVTKGeometryPainterBuilder::Attribute<string>("name") +
                                           KGVTKGeometryPainterBuilder::Attribute<string>("file") +
                                           KGVTKGeometryPainterBuilder::Attribute<string>("path") +
                                           KGVTKGeometryPainterBuilder::Attribute<string>("write_stl") +
                                           KGVTKGeometryPainterBuilder::Attribute<string>("surfaces") +
                                           KGVTKGeometryPainterBuilder::Attribute<string>("spaces");

STATICINT sKGVTKGeometryPainterWindow = KVTKWindowBuilder::ComplexElement<KGVTKGeometryPainter>("vtk_geometry_painter");

}  // namespace katrin
