#include "KGVTKMeshPainterBuilder.hh"
#include "KVTKWindowBuilder.h"

using namespace KGeoBag;
using namespace std;

namespace katrin
{

    STATICINT sKGVTKMeshPainterStructure =
        KGVTKMeshPainterBuilder::Attribute< string >( "name" ) +
        KGVTKMeshPainterBuilder::Attribute< string >( "file" ) +
        KGVTKMeshPainterBuilder::Attribute< unsigned int >( "arc_count" ) +
        KGVTKMeshPainterBuilder::Attribute< string >( "color_mode" ) +
        KGVTKMeshPainterBuilder::Attribute< string >( "surfaces" ) +
        KGVTKMeshPainterBuilder::Attribute< string >( "spaces" );

    STATICINT sKGVTKMeshPainter =
        KVTKWindowBuilder::ComplexElement< KGVTKMeshPainter >( "vtk_mesh_painter" );

}
