#ifndef Kassiopeia_KSTrajExactTypes_h_
#define Kassiopeia_KSTrajExactTypes_h_

#include "KSMathControl.h"
#include "KSMathDifferentiator.h"
#include "KSMathIntegrator.h"
#include "KSMathInterpolator.h"
#include "KSMathSystem.h"
#include "KSTrajExactDerivative.h"
#include "KSTrajExactError.h"
#include "KSTrajExactParticle.h"

namespace Kassiopeia
{

typedef KSMathSystem<KSTrajExactParticle, KSTrajExactDerivative, KSTrajExactError> KSTrajExactSystem;
typedef KSMathControl<KSTrajExactSystem> KSTrajExactControl;
typedef KSMathDifferentiator<KSTrajExactSystem> KSTrajExactDifferentiator;
typedef KSMathIntegrator<KSTrajExactSystem> KSTrajExactIntegrator;
typedef KSMathInterpolator<KSTrajExactSystem> KSTrajExactInterpolator;

}  // namespace Kassiopeia

#endif
