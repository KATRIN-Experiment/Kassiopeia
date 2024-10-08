//
// Created by trost on 25.07.16.
//

#include "KROOTPadBuilder.h"

#include "KElementProcessor.hh"

using namespace std;

namespace katrin
{

STATICINT sKROOTPadStructure =
    KROOTPadBuilder::Attribute<std::string>("name") +
    KROOTPadBuilder::Attribute<double>("xlow") +
    KROOTPadBuilder::Attribute<double>("ylow") +
    KROOTPadBuilder::Attribute<double>("xup") +
    KROOTPadBuilder::Attribute<double>("yup") +
    KROOTPadBuilder::Attribute<double>("xmin") +
    KROOTPadBuilder::Attribute<double>("ymin") +
    KROOTPadBuilder::Attribute<double>("xmax") +
    KROOTPadBuilder::Attribute<double>("ymax");


}
