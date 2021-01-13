#ifndef Kassiopeia_KSTrajAdiabaticSpinTypes_h_
#define Kassiopeia_KSTrajAdiabaticSpinTypes_h_

#include "KSMathControl.h"
#include "KSMathDifferentiator.h"
#include "KSMathIntegrator.h"
#include "KSMathInterpolator.h"
#include "KSMathSystem.h"
#include "KSTrajAdiabaticSpinDerivative.h"
#include "KSTrajAdiabaticSpinError.h"
#include "KSTrajAdiabaticSpinParticle.h"

namespace Kassiopeia
{

typedef KSMathSystem<KSTrajAdiabaticSpinParticle, KSTrajAdiabaticSpinDerivative, KSTrajAdiabaticSpinError>
    KSTrajAdiabaticSpinSystem;
using KSTrajAdiabaticSpinControl = KSMathControl<KSTrajAdiabaticSpinSystem>;
using KSTrajAdiabaticSpinDifferentiator = KSMathDifferentiator<KSTrajAdiabaticSpinSystem>;
using KSTrajAdiabaticSpinIntegrator = KSMathIntegrator<KSTrajAdiabaticSpinSystem>;
using KSTrajAdiabaticSpinInterpolator = KSMathInterpolator<KSTrajAdiabaticSpinSystem>;

}  // namespace Kassiopeia

#endif
