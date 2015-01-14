#include "KGVTKMeshPainterBuilder.hh"
#include "KVTKWindow.h"

using namespace KGeoBag;
namespace katrin
{

    static const int sKGVTKMeshPainterStructure =
        KGVTKMeshPainterBuilder::Attribute< string >( "name" ) +
        KGVTKMeshPainterBuilder::Attribute< string >( "file" ) +
        KGVTKMeshPainterBuilder::Attribute< unsigned int >( "arc_count" ) +
        KGVTKMeshPainterBuilder::Attribute< string >( "color_mode" ) +
        KGVTKMeshPainterBuilder::Attribute< string >( "surfaces" ) +
        KGVTKMeshPainterBuilder::Attribute< string >( "spaces" );

    static const int sKGVTKMeshPainter =
        KVTKWindowBuilder::ComplexElement< KGVTKMeshPainter >( "vtk_mesh_painter" );

}
