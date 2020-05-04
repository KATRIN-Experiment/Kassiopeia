#include "KGROOTGeometryPainterBuilder.hh"

#include "KROOTPadBuilder.h"
#include "KROOTWindowBuilder.h"

using namespace KGeoBag;
using namespace std;

namespace katrin
{

STATICINT sKGROOTGeometryPainterStructure = KGROOTGeometryPainterBuilder::Attribute<string>("name") +
                                            KGROOTGeometryPainterBuilder::Attribute<string>("surfaces") +
                                            KGROOTGeometryPainterBuilder::Attribute<string>("spaces") +
                                            KGROOTGeometryPainterBuilder::Attribute<KThreeVector>("plane_normal") +
                                            KGROOTGeometryPainterBuilder::Attribute<KThreeVector>("plane_point") +
                                            KGROOTGeometryPainterBuilder::Attribute<bool>("swap_axis") +
                                            KGROOTGeometryPainterBuilder::Attribute<double>("epsilon");


STATICINT sKGROOTGeometryPainterWindow =
    KROOTWindowBuilder::ComplexElement<KGROOTGeometryPainter>("root_geometry_painter");

STATICINT sKGROOTGeometryPainterPad = KROOTPadBuilder::ComplexElement<KGROOTGeometryPainter>("root_geometry_painter");

}  // namespace katrin
