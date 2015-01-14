#include "KGROOTGeometryPainterBuilder.hh"
#include "KROOTWindow.h"
#include "KROOTPad.h"

using namespace KGeoBag;
namespace katrin
{

    static const int sKGROOTGeometryPainterStructure =
        KGROOTGeometryPainterBuilder::Attribute< string >( "name" ) +
        KGROOTGeometryPainterBuilder::Attribute< string >( "surfaces" ) +
        KGROOTGeometryPainterBuilder::Attribute< string >( "spaces" ) +
        KGROOTGeometryPainterBuilder::Attribute< KThreeVector >( "plane_normal" ) +
        KGROOTGeometryPainterBuilder::Attribute< KThreeVector >( "plane_point" ) +
        KGROOTGeometryPainterBuilder::Attribute< bool >( "swap_axis" );

    static const int sKGROOTGeometryPainterWindow =
        KROOTWindowBuilder::ComplexElement< KGROOTGeometryPainter >( "root_geometry_painter" );

    static const int sKGROOTGeometryPainterPad =
		KROOTPadBuilder::ComplexElement< KGROOTGeometryPainter >( "root_geometry_painter" );

}
