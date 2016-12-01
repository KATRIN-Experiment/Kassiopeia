//
// Created by trost on 25.07.16.
//

#include "KROOTWindowBuilder.h"
#include "KRoot.h"
#include "KROOTPadBuilder.h"
#include "KElementProcessor.hh"

using namespace std;

namespace katrin
{

STATICINT sKROOTWindowStructure =
    KROOTWindowBuilder::Attribute< string >( "name" ) +
    KROOTWindowBuilder::Attribute< unsigned int >( "canvas_width" ) +
    KROOTWindowBuilder::Attribute< unsigned int >( "canvas_height" ) +
    KROOTWindowBuilder::Attribute< bool >( "active" ) +
    KROOTWindowBuilder::Attribute< bool >( "write_enabled" ) +
    KROOTWindowBuilder::Attribute< string >( "path" ) +
    KROOTWindowBuilder::ComplexElement< KROOTPad >( "root_pad" );

STATICINT sKROOTWindow =
    KRootBuilder::ComplexElement< KROOTWindow >( "root_window" );

STATICINT sKROOTWindowCompat =
    KElementProcessor::ComplexElement< KROOTWindow >( "root_window" );

}
