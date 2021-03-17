#include "KSROOTTrackPainterBuilder.h"

#include "KROOTPad.h"
#include "KROOTPadBuilder.h"
#include "KROOTWindowBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

STATICINT sKSROOTTrackPainterStructure = KSROOTTrackPainterBuilder::Attribute<std::string>("name") +
                                         KSROOTTrackPainterBuilder::Attribute<std::string>("path") +
                                         KSROOTTrackPainterBuilder::Attribute<std::string>("base") +
                                         KSROOTTrackPainterBuilder::Attribute<std::string>("x_axis") +
                                         KSROOTTrackPainterBuilder::Attribute<std::string>("y_axis") +
                                         KSROOTTrackPainterBuilder::Attribute<KThreeVector>("plane_normal") +
                                         KSROOTTrackPainterBuilder::Attribute<KThreeVector>("plane_point") +
                                         KSROOTTrackPainterBuilder::Attribute<bool>("swap_axis") +
                                         KSROOTTrackPainterBuilder::Attribute<double>("epsilon") +
                                         KSROOTTrackPainterBuilder::Attribute<std::string>("step_output_group_name") +
                                         KSROOTTrackPainterBuilder::Attribute<std::string>("position_name") +
                                         KSROOTTrackPainterBuilder::Attribute<std::string>("track_output_group_name") +
                                         KSROOTTrackPainterBuilder::Attribute<std::string>("color_variable") +
                                         KSROOTTrackPainterBuilder::Attribute<std::string>("color_mode") +
                                         KSROOTTrackPainterBuilder::Attribute<std::string>("color_palette") +
                                         KSROOTTrackPainterBuilder::Attribute<std::string>("color") +
                                         KSROOTTrackPainterBuilder::Attribute<std::string>("add_color") +
                                         KSROOTTrackPainterBuilder::Attribute<std::string>("draw_options") +
                                         KSROOTTrackPainterBuilder::Attribute<std::string>("plot_mode") +
                                         KSROOTTrackPainterBuilder::Attribute<bool>("axial_mirror");


STATICINT sKSROOTTrackPainterWindow = KROOTWindowBuilder::ComplexElement<KSROOTTrackPainter>("root_track_painter");

STATICINT sKSROOTTrackPainterPad = KROOTPadBuilder::ComplexElement<KSROOTTrackPainter>("root_track_painter");

}  // namespace katrin
