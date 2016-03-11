#include "KGVTKGeometryPainterBuilder.hh"
#include "KVTKWindow.h"

using namespace KGeoBag;
namespace katrin
{

    STATICINT sKGVTKGeometryPainterStructure =
        KGVTKGeometryPainterBuilder::Attribute< string >( "name" ) +
        KGVTKGeometryPainterBuilder::Attribute< string >( "file" ) +
        KGVTKGeometryPainterBuilder::Attribute< string >( "path" ) +
        KGVTKGeometryPainterBuilder::Attribute< string >( "write_stl" ) +
        KGVTKGeometryPainterBuilder::Attribute< string >( "surfaces" ) +
        KGVTKGeometryPainterBuilder::Attribute< string >( "spaces" );

    STATICINT sKGVTKGeometryPainterWindow =
        KVTKWindowBuilder::ComplexElement< KGVTKGeometryPainter >( "vtk_geometry_painter" );

}
