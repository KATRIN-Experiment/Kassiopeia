//
// Created by trost on 25.07.16.
//

#include "KROOTWindowBuilder.h"

#include "KElementProcessor.hh"
#include "KROOTPadBuilder.h"
#include "KRoot.h"

using namespace std;

namespace katrin
{

STATICINT sKROOTWindowStructure =
    KROOTWindowBuilder::Attribute<std::string>("name") + KROOTWindowBuilder::Attribute<unsigned int>("canvas_width") +
    KROOTWindowBuilder::Attribute<unsigned int>("canvas_height") + KROOTWindowBuilder::Attribute<bool>("active") +
    KROOTWindowBuilder::Attribute<bool>("write_enabled") + KROOTWindowBuilder::Attribute<std::string>("path") +
    KROOTWindowBuilder::ComplexElement<KROOTPad>("root_pad");

STATICINT sKROOTWindow = KRootBuilder::ComplexElement<KROOTWindow>("root_window");

STATICINT sKROOTWindowCompat = KElementProcessor::ComplexElement<KROOTWindow>("root_window");

}  // namespace katrin
