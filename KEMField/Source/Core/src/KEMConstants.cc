#include "KEMConstants.hh"

#include "KConst.h"

#include <cmath>

namespace KEMField
{
const double KEMConstants::Pi = katrin::KConst::Pi();
const double KEMConstants::PiOverTwo = katrin::KConst::Pi() / 2.;
const double KEMConstants::Eps0 = katrin::KConst::EpsNull();
const double KEMConstants::OneOverFourPiEps0 = 1. / katrin::KConst::FourPiEps();
const double KEMConstants::Mu0 = katrin::KConst::MuNull();
const double KEMConstants::Mu0OverPi = 4. * 1.e-7;
}  // namespace KEMField
