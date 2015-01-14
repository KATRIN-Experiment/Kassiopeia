#include "KEMConstants.hh"

#include <cmath>

#ifdef KEMFIELD_USE_KOMMON

#include "KConst.h"

namespace KEMField
{
  const double KEMConstants::Pi = katrin::KConst::Pi();
  const double KEMConstants::PiOverTwo = katrin::KConst::Pi()/2.;
  const double KEMConstants::Eps0 = katrin::KConst::EpsNull();
  const double KEMConstants::Mu0 = katrin::KConst::MuNull();
  const double KEMConstants::Mu0OverPi = 4.*1.e-7;
}

#else

namespace KEMField
{
  const double KEMConstants::Pi = M_PI;
  const double KEMConstants::PiOverTwo = M_PI/2.;
  const double KEMConstants::Eps0 = 8.85418782e-12;
  const double KEMConstants::Mu0 = 1.25663706e-06;
  const double KEMConstants::Mu0OverPi = 4.*1.e-7;
}
#endif
