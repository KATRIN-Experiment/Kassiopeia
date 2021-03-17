#include "KGVTKAxialMeshPainterBuilder.hh"

#include "KVTKWindowBuilder.h"

using namespace KGeoBag;
using namespace std;

namespace katrin
{

STATICINT sKGVTKAxialMeshPainterStructure = KGVTKAxialMeshPainterBuilder::Attribute<std::string>("name") +
                                            KGVTKAxialMeshPainterBuilder::Attribute<std::string>("file") +
                                            KGVTKAxialMeshPainterBuilder::Attribute<unsigned int>("arc_count") +
                                            KGVTKAxialMeshPainterBuilder::Attribute<std::string>("color_mode") +
                                            KGVTKAxialMeshPainterBuilder::Attribute<std::string>("surfaces") +
                                            KGVTKAxialMeshPainterBuilder::Attribute<std::string>("spaces");

STATICINT sKGVTKAxialMeshPainterWindow =
    KVTKWindowBuilder::ComplexElement<KGVTKAxialMeshPainter>("vtk_axial_mesh_painter");

}  // namespace katrin
