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

using KSTrajExactSystem = KSMathSystem<KSTrajExactParticle, KSTrajExactDerivative, KSTrajExactError>;
using KSTrajExactControl = KSMathControl<KSTrajExactSystem>;
using KSTrajExactDifferentiator = KSMathDifferentiator<KSTrajExactSystem>;
using KSTrajExactIntegrator = KSMathIntegrator<KSTrajExactSystem>;
using KSTrajExactInterpolator = KSMathInterpolator<KSTrajExactSystem>;

}  // namespace Kassiopeia

#endif
