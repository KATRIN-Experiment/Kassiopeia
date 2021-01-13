#ifndef Kassiopeia_KSTrajExactTrappedTypes_h_
#define Kassiopeia_KSTrajExactTrappedTypes_h_

#include "KSMathControl.h"
#include "KSMathDifferentiator.h"
#include "KSMathIntegrator.h"
#include "KSMathInterpolator.h"
#include "KSMathSystem.h"
#include "KSTrajExactTrappedDerivative.h"
#include "KSTrajExactTrappedError.h"
#include "KSTrajExactTrappedParticle.h"

namespace Kassiopeia
{

typedef KSMathSystem<KSTrajExactTrappedParticle, KSTrajExactTrappedDerivative, KSTrajExactTrappedError>
    KSTrajExactTrappedSystem;
using KSTrajExactTrappedControl = KSMathControl<KSTrajExactTrappedSystem>;
using KSTrajExactTrappedDifferentiator = KSMathDifferentiator<KSTrajExactTrappedSystem>;
using KSTrajExactTrappedIntegrator = KSMathIntegrator<KSTrajExactTrappedSystem>;
using KSTrajExactTrappedInterpolator = KSMathInterpolator<KSTrajExactTrappedSystem>;

}  // namespace Kassiopeia

#endif
