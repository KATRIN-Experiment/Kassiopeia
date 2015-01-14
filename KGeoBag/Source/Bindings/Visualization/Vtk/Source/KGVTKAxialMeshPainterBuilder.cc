#include "KGVTKAxialMeshPainterBuilder.hh"
#include "KVTKWindow.h"

using namespace KGeoBag;
namespace katrin
{

    static const int sKGVTKAxialMeshPainterStructure =
        KGVTKAxialMeshPainterBuilder::Attribute< string >( "name" ) +
        KGVTKAxialMeshPainterBuilder::Attribute< string >( "file" ) +
        KGVTKAxialMeshPainterBuilder::Attribute< unsigned int >( "arc_count" ) +
        KGVTKAxialMeshPainterBuilder::Attribute< string >( "color_mode" ) +
        KGVTKAxialMeshPainterBuilder::Attribute< string >( "surfaces" ) +
        KGVTKAxialMeshPainterBuilder::Attribute< string >( "spaces" );

    static const int sKGVTKAxialMeshPainterWindow =
        KVTKWindowBuilder::ComplexElement< KGVTKAxialMeshPainter >( "vtk_axial_mesh_painter" );

}
