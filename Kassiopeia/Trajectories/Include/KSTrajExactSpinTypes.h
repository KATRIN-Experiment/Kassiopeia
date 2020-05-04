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
typedef KSMathControl<KSTrajExactSpinSystem> KSTrajExactSpinControl;
typedef KSMathDifferentiator<KSTrajExactSpinSystem> KSTrajExactSpinDifferentiator;
typedef KSMathIntegrator<KSTrajExactSpinSystem> KSTrajExactSpinIntegrator;
typedef KSMathInterpolator<KSTrajExactSpinSystem> KSTrajExactSpinInterpolator;

}  // namespace Kassiopeia

#endif
