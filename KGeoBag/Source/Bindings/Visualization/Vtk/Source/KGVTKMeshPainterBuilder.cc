#include "KGVTKMeshPainterBuilder.hh"

#include "KVTKWindowBuilder.h"

using namespace KGeoBag;
using namespace std;

namespace katrin
{

STATICINT sKGVTKMeshPainterStructure = KGVTKMeshPainterBuilder::Attribute<std::string>("name") +
                                       KGVTKMeshPainterBuilder::Attribute<std::string>("file") +
                                       KGVTKMeshPainterBuilder::Attribute<unsigned int>("arc_count") +
                                       KGVTKMeshPainterBuilder::Attribute<std::string>("color_mode") +
                                       KGVTKMeshPainterBuilder::Attribute<std::string>("surfaces") +
                                       KGVTKMeshPainterBuilder::Attribute<std::string>("spaces");

STATICINT sKGVTKMeshPainter = KVTKWindowBuilder::ComplexElement<KGVTKMeshPainter>("vtk_mesh_painter");

}  // namespace katrin
