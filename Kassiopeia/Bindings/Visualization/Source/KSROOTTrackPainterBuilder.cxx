#include "KSROOTTrackPainterBuilder.h"
#include "KROOTWindow.h"
#include "KROOTPad.h"

using namespace Kassiopeia;
namespace katrin
{

    STATICINT sKSROOTTrackPainterStructure =
    	KSROOTTrackPainterBuilder::Attribute< string >( "name" ) +
    	KSROOTTrackPainterBuilder::Attribute< string >( "path" ) +
    	KSROOTTrackPainterBuilder::Attribute< string >( "base" ) +
    	KSROOTTrackPainterBuilder::Attribute< string >( "x_axis" ) +
    	KSROOTTrackPainterBuilder::Attribute< string >( "y_axis" ) +
    	KSROOTTrackPainterBuilder::Attribute< string >( "step_output_group_name" ) +
    	KSROOTTrackPainterBuilder::Attribute< string >( "position_name" ) +
    	KSROOTTrackPainterBuilder::Attribute< string >( "track_output_group_name" ) +
    	KSROOTTrackPainterBuilder::Attribute< string >( "color_variable" ) +
    	KSROOTTrackPainterBuilder::Attribute< string >( "color_mode" ) +
    	KSROOTTrackPainterBuilder::Attribute< string >( "color_palette" ) +
    	KSROOTTrackPainterBuilder::Attribute< string >( "color" ) +
    	KSROOTTrackPainterBuilder::Attribute< string >( "add_color" ) +
    	KSROOTTrackPainterBuilder::Attribute< string >( "draw_options" ) +
    	KSROOTTrackPainterBuilder::Attribute< string >( "plot_mode" ) +
    	KSROOTTrackPainterBuilder::Attribute< bool >( "axial_mirror" );



    STATICINT sKSROOTTrackPainterWindow =
		KROOTWindowBuilder::ComplexElement< KSROOTTrackPainter >( "root_track_painter" );

    STATICINT sKSROOTTrackPainterPad =
		KROOTPadBuilder::ComplexElement< KSROOTTrackPainter >( "root_track_painter" );

}


