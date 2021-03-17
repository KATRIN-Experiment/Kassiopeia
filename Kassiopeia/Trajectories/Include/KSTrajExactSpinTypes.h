#ifndef Kassiopeia_KSTrajExactSpinTypes_h_
#define Kassiopeia_KSTrajExactSpinTypes_h_

#include "KSMathControl.h"
#include "KSMathDifferentiator.h"
#include "KSMathIntegrator.h"
#include "KSMathInterpolator.h"
#include "KSMathSystem.h"
#include "KSTrajExactSpinDerivative.h"
#include "KSTrajExactSpinError.h"
#include "KSTrajExactSpinParticle.h"

namespace Kassiopeia
{

typedef KSMathSystem<KSTrajExactSpinParticle, KSTrajExactSpinDerivative, KSTrajExactSpinError> KSTrajExactSpinSystem;
using KSTrajExactSpinControl = KSMathControl<KSTrajExactSpinSystem>;
using KSTrajExactSpinDifferentiator = KSMathDifferentiator<KSTrajExactSpinSystem>;
using KSTrajExactSpinIntegrator = KSMathIntegrator<KSTrajExactSpinSystem>;
using KSTrajExactSpinInterpolator = KSMathInterpolator<KSTrajExactSpinSystem>;

}  // namespace Kassiopeia

#endif
