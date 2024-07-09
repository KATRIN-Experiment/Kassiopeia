#ifndef Kassiopeia_KSTrajAdiabaticTypes_h_
#define Kassiopeia_KSTrajAdiabaticTypes_h_

#include "KSMathControl.h"
#include "KSMathDifferentiator.h"
#include "KSMathIntegrator.h"
#include "KSMathInterpolator.h"
#include "KSMathSystem.h"
#include "KSTrajAdiabaticDerivative.h"
#include "KSTrajAdiabaticError.h"
#include "KSTrajAdiabaticParticle.h"

namespace Kassiopeia
{

using KSTrajAdiabaticSystem = KSMathSystem<KSTrajAdiabaticParticle, KSTrajAdiabaticDerivative, KSTrajAdiabaticError>;
using KSTrajAdiabaticControl = KSMathControl<KSTrajAdiabaticSystem>;
using KSTrajAdiabaticDifferentiator = KSMathDifferentiator<KSTrajAdiabaticSystem>;
using KSTrajAdiabaticIntegrator = KSMathIntegrator<KSTrajAdiabaticSystem>;
using KSTrajAdiabaticInterpolator = KSMathInterpolator<KSTrajAdiabaticSystem>;

}  // namespace Kassiopeia

#endif
