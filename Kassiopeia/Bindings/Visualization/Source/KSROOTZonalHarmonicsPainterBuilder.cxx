#include "KSROOTZonalHarmonicsPainterBuilder.h"

#include "KROOTPadBuilder.h"
#include "KROOTWindowBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

STATICINT sKSROOTZonalHarmonicsPainterStructure =
    KSROOTZonalHarmonicsPainterBuilder::Attribute<std::string>("name") +
    KSROOTZonalHarmonicsPainterBuilder::Attribute<std::string>("x_axis") +
    KSROOTZonalHarmonicsPainterBuilder::Attribute<std::string>("y_axis") +
    KSROOTZonalHarmonicsPainterBuilder::Attribute<std::string>("electric_field") +
    KSROOTZonalHarmonicsPainterBuilder::Attribute<std::string>("magnetic_field") +
    KSROOTZonalHarmonicsPainterBuilder::Attribute<double>("r_min") +
    KSROOTZonalHarmonicsPainterBuilder::Attribute<double>("r_max") +
    KSROOTZonalHarmonicsPainterBuilder::Attribute<double>("z_min") +
    KSROOTZonalHarmonicsPainterBuilder::Attribute<double>("z_max") +
    KSROOTZonalHarmonicsPainterBuilder::Attribute<double>("z_dist") +
    KSROOTZonalHarmonicsPainterBuilder::Attribute<double>("r_dist") +
    KSROOTZonalHarmonicsPainterBuilder::Attribute<int>("r_steps") +
    KSROOTZonalHarmonicsPainterBuilder::Attribute<int>("z_steps") +
    KSROOTZonalHarmonicsPainterBuilder::Attribute<std::string>("path") +
    KSROOTZonalHarmonicsPainterBuilder::Attribute<std::string>("file") +
    KSROOTZonalHarmonicsPainterBuilder::Attribute<bool>("write") +
    KSROOTZonalHarmonicsPainterBuilder::Attribute<bool>("draw_source_points") +
    KSROOTZonalHarmonicsPainterBuilder::Attribute<bool>("draw_convergence_area") +
    KSROOTZonalHarmonicsPainterBuilder::Attribute<bool>("draw_central_boundary") +
    KSROOTZonalHarmonicsPainterBuilder::Attribute<bool>("draw_remote_boundary");
//KSROOTZonalHarmonicsPainterBuilder::Attribute< string >("geometry_type" ) +
//KSROOTZonalHarmonicsPainterBuilder::Attribute< double >( "radial_safety_margin" );


STATICINT sKSROOTZonalHarmonicsPainterWindow =
    KROOTWindowBuilder::ComplexElement<KSROOTZonalHarmonicsPainter>("root_zh_painter");

STATICINT sKSROOTZonalHarmonicsPainterPad =
    KROOTPadBuilder::ComplexElement<KSROOTZonalHarmonicsPainter>("root_zh_painter");

}  // namespace katrin
