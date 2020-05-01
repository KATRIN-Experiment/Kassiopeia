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

typedef KSMathSystem<KSTrajAdiabaticParticle, KSTrajAdiabaticDerivative, KSTrajAdiabaticError> KSTrajAdiabaticSystem;
typedef KSMathControl<KSTrajAdiabaticSystem> KSTrajAdiabaticControl;
typedef KSMathDifferentiator<KSTrajAdiabaticSystem> KSTrajAdiabaticDifferentiator;
typedef KSMathIntegrator<KSTrajAdiabaticSystem> KSTrajAdiabaticIntegrator;
typedef KSMathInterpolator<KSTrajAdiabaticSystem> KSTrajAdiabaticInterpolator;

}  // namespace Kassiopeia

#endif
