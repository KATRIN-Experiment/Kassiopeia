#include "KSROOTZonalHarmonicsPainterBuilder.h"

#include "KROOTPadBuilder.h"
#include "KROOTWindowBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

STATICINT sKSROOTZonalHarmonicsPainterStructure =
    KSROOTZonalHarmonicsPainterBuilder::Attribute<string>("name") +
    KSROOTZonalHarmonicsPainterBuilder::Attribute<string>("x_axis") +
    KSROOTZonalHarmonicsPainterBuilder::Attribute<string>("y_axis") +
    KSROOTZonalHarmonicsPainterBuilder::Attribute<string>("electric_field") +
    KSROOTZonalHarmonicsPainterBuilder::Attribute<string>("magnetic_field") +
    KSROOTZonalHarmonicsPainterBuilder::Attribute<double>("r_min") +
    KSROOTZonalHarmonicsPainterBuilder::Attribute<double>("r_max") +
    KSROOTZonalHarmonicsPainterBuilder::Attribute<double>("z_min") +
    KSROOTZonalHarmonicsPainterBuilder::Attribute<double>("z_max") +
    KSROOTZonalHarmonicsPainterBuilder::Attribute<double>("z_dist") +
    KSROOTZonalHarmonicsPainterBuilder::Attribute<double>("r_dist") +
    KSROOTZonalHarmonicsPainterBuilder::Attribute<int>("r_steps") +
    KSROOTZonalHarmonicsPainterBuilder::Attribute<int>("z_steps") +
    KSROOTZonalHarmonicsPainterBuilder::Attribute<string>("path") +
    KSROOTZonalHarmonicsPainterBuilder::Attribute<string>("file") +
    KSROOTZonalHarmonicsPainterBuilder::Attribute<bool>("write");
//KSROOTZonalHarmonicsPainterBuilder::Attribute< string >("geometry_type" ) +
//KSROOTZonalHarmonicsPainterBuilder::Attribute< double >( "radial_safety_margin" );


STATICINT sKSROOTZonalHarmonicsPainterWindow =
    KROOTWindowBuilder::ComplexElement<KSROOTZonalHarmonicsPainter>("root_zh_painter");

STATICINT sKSROOTZonalHarmonicsPainterPad =
    KROOTPadBuilder::ComplexElement<KSROOTZonalHarmonicsPainter>("root_zh_painter");

}  // namespace katrin
