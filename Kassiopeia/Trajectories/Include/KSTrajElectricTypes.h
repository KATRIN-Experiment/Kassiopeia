#ifndef Kassiopeia_KSTrajElectricTypes_h_
#define Kassiopeia_KSTrajElectricTypes_h_

#include "KSMathControl.h"
#include "KSMathDifferentiator.h"
#include "KSMathIntegrator.h"
#include "KSMathInterpolator.h"
#include "KSMathSystem.h"
#include "KSTrajElectricDerivative.h"
#include "KSTrajElectricError.h"
#include "KSTrajElectricParticle.h"

namespace Kassiopeia
{

typedef KSMathSystem<KSTrajElectricParticle, KSTrajElectricDerivative, KSTrajElectricError> KSTrajElectricSystem;
using KSTrajElectricControl = KSMathControl<KSTrajElectricSystem>;
using KSTrajElectricDifferentiator = KSMathDifferentiator<KSTrajElectricSystem>;
using KSTrajElectricIntegrator = KSMathIntegrator<KSTrajElectricSystem>;
using KSTrajElectricInterpolator = KSMathInterpolator<KSTrajElectricSystem>;

}  // namespace Kassiopeia

#endif
