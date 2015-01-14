#include "KSROOTPotentialPainterBuilder.h"
#include "KROOTWindow.h"
#include "KROOTPad.h"

using namespace Kassiopeia;
namespace katrin
{

    static int sKSROOTPotentialPainterStructure =
        KSROOTPotentialPainterBuilder::Attribute< string >( "name" ) +
        KSROOTPotentialPainterBuilder::Attribute< string >( "x_axis" ) +
        KSROOTPotentialPainterBuilder::Attribute< string >( "y_axis" ) +
        KSROOTPotentialPainterBuilder::Attribute< string >( "electric_field" ) +
        KSROOTPotentialPainterBuilder::Attribute< double >( "r_max" ) +
        KSROOTPotentialPainterBuilder::Attribute< double >( "z_min" ) +
        KSROOTPotentialPainterBuilder::Attribute< double >( "z_max" ) +
        KSROOTPotentialPainterBuilder::Attribute< int >( "r_steps" ) +
        KSROOTPotentialPainterBuilder::Attribute< int >( "z_steps" ) +
        KSROOTPotentialPainterBuilder::Attribute< bool >( "calc_pot" );


    static int sKSROOTPotentialPainterWindow =
        KROOTWindowBuilder::ComplexElement< KSROOTPotentialPainter >( "root_potential_painter" );

    static int sKSROOTPotentialPainterPad =
        KROOTPadBuilder::ComplexElement< KSROOTPotentialPainter >( "root_potential_painter" );

}

