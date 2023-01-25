#ifndef Kassiopeia_KSTrajMagneticTypes_h_
#define Kassiopeia_KSTrajMagneticTypes_h_

#include "KSMathControl.h"
#include "KSMathDifferentiator.h"
#include "KSMathIntegrator.h"
#include "KSMathInterpolator.h"
#include "KSMathSystem.h"
#include "KSTrajMagneticDerivative.h"
#include "KSTrajMagneticError.h"
#include "KSTrajMagneticParticle.h"

namespace Kassiopeia
{

using KSTrajMagneticSystem = KSMathSystem<KSTrajMagneticParticle, KSTrajMagneticDerivative, KSTrajMagneticError>;
using KSTrajMagneticControl = KSMathControl<KSTrajMagneticSystem>;
using KSTrajMagneticDifferentiator = KSMathDifferentiator<KSTrajMagneticSystem>;
using KSTrajMagneticIntegrator = KSMathIntegrator<KSTrajMagneticSystem>;
using KSTrajMagneticInterpolator = KSMathInterpolator<KSTrajMagneticSystem>;

}  // namespace Kassiopeia

#endif
