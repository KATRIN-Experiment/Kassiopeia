#include "KSROOTPotentialPainterBuilder.h"
#include "KROOTWindowBuilder.h"
#include "KROOTPadBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

    STATICINT sKSROOTPotentialPainterStructure =
        KSROOTPotentialPainterBuilder::Attribute< string >( "name" ) +
        KSROOTPotentialPainterBuilder::Attribute< string >( "x_axis" ) +
        KSROOTPotentialPainterBuilder::Attribute< string >( "y_axis" ) +
        KSROOTPotentialPainterBuilder::Attribute< string >( "electric_field" ) +
        KSROOTPotentialPainterBuilder::Attribute< double >( "r_max" ) +
        KSROOTPotentialPainterBuilder::Attribute< double >( "z_min" ) +
        KSROOTPotentialPainterBuilder::Attribute< double >( "z_max" ) +
        KSROOTPotentialPainterBuilder::Attribute< int >( "r_steps" ) +
        KSROOTPotentialPainterBuilder::Attribute< int >( "z_steps" ) +
        KSROOTPotentialPainterBuilder::Attribute< bool >( "calc_pot" ) +
        KSROOTPotentialPainterBuilder::Attribute< bool >( "compare_fields" ) +
        KSROOTPotentialPainterBuilder::Attribute< string >( "reference_field" );


    STATICINT sKSROOTPotentialPainterWindow =
        KROOTWindowBuilder::ComplexElement< KSROOTPotentialPainter >( "root_potential_painter" );

    STATICINT sKSROOTPotentialPainterPad =
        KROOTPadBuilder::ComplexElement< KSROOTPotentialPainter >( "root_potential_painter" );

}

