#include "KSROOTMagFieldPainterBuilder.h"

#include "KROOTPadBuilder.h"
#include "KROOTWindowBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

STATICINT sKSROOTMagFieldPainterStructure =
    KSROOTMagFieldPainterBuilder::Attribute<std::string>("name") +
    KSROOTMagFieldPainterBuilder::Attribute<std::string>("x_axis") +
    KSROOTMagFieldPainterBuilder::Attribute<std::string>("y_axis") +
    KSROOTMagFieldPainterBuilder::Attribute<std::string>("magnetic_field") +
    KSROOTMagFieldPainterBuilder::Attribute<double>("r_max") +
    KSROOTMagFieldPainterBuilder::Attribute<double>("z_min") +
    KSROOTMagFieldPainterBuilder::Attribute<double>("z_max") + KSROOTMagFieldPainterBuilder::Attribute<int>("r_steps") +
    KSROOTMagFieldPainterBuilder::Attribute<int>("z_steps") +
    KSROOTMagFieldPainterBuilder::Attribute<std::string>("plot") +
    KSROOTMagFieldPainterBuilder::Attribute<bool>("z_axis_logscale") +
    KSROOTMagFieldPainterBuilder::Attribute<bool>("magnetic_gradient_numerical") +
    KSROOTMagFieldPainterBuilder::Attribute<std::string>("draw") +
    KSROOTMagFieldPainterBuilder::Attribute<bool>("axial_symmetry") +
    KSROOTMagFieldPainterBuilder::Attribute<double>("z_fix");


STATICINT sKSROOTMagFieldPainterWindow =
    KROOTWindowBuilder::ComplexElement<KSROOTMagFieldPainter>("root_magfield_painter");

STATICINT sKSROOTMagFieldPainterPad = KROOTPadBuilder::ComplexElement<KSROOTMagFieldPainter>("root_magfield_painter");

}  // namespace katrin
