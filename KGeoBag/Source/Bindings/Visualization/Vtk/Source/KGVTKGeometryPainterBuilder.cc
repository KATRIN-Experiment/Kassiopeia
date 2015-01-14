#include "KGVTKGeometryPainterBuilder.hh"
#include "KVTKWindow.h"

using namespace KGeoBag;
namespace katrin
{

    static const int sKGVTKGeometryPainterStructure =
        KGVTKGeometryPainterBuilder::Attribute< string >( "name" ) +
        KGVTKGeometryPainterBuilder::Attribute< string >( "file" ) +
        KGVTKGeometryPainterBuilder::Attribute< string >( "path" ) +
        KGVTKGeometryPainterBuilder::Attribute< string >( "surfaces" ) +
        KGVTKGeometryPainterBuilder::Attribute< string >( "spaces" );

    static const int sKGVTKGeometryPainterWindow =
        KVTKWindowBuilder::ComplexElement< KGVTKGeometryPainter >( "vtk_geometry_painter" );

}
