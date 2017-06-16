//
// Created by trost on 25.07.16.
//

#include "KElementProcessor.hh"
#include "KROOTPadBuilder.h"

using namespace std;

namespace katrin
{

STATICINT sKROOTPadStructure =
        KROOTPadBuilder::Attribute< string >( "name" ) +
        KROOTPadBuilder::Attribute< double >( "xlow" ) +
        KROOTPadBuilder::Attribute< double >( "ylow" ) +
        KROOTPadBuilder::Attribute< double >( "xup" ) +
        KROOTPadBuilder::Attribute< double >( "yup" );


}
