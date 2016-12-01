#include "KGVTKAxialMeshPainterBuilder.hh"
#include "KVTKWindowBuilder.h"

using namespace KGeoBag;
using namespace std;

namespace katrin
{

    STATICINT sKGVTKAxialMeshPainterStructure =
        KGVTKAxialMeshPainterBuilder::Attribute< string >( "name" ) +
        KGVTKAxialMeshPainterBuilder::Attribute< string >( "file" ) +
        KGVTKAxialMeshPainterBuilder::Attribute< unsigned int >( "arc_count" ) +
        KGVTKAxialMeshPainterBuilder::Attribute< string >( "color_mode" ) +
        KGVTKAxialMeshPainterBuilder::Attribute< string >( "surfaces" ) +
        KGVTKAxialMeshPainterBuilder::Attribute< string >( "spaces" );

    STATICINT sKGVTKAxialMeshPainterWindow =
        KVTKWindowBuilder::ComplexElement< KGVTKAxialMeshPainter >( "vtk_axial_mesh_painter" );

}
