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
typedef KSMathControl<KSTrajElectricSystem> KSTrajElectricControl;
typedef KSMathDifferentiator<KSTrajElectricSystem> KSTrajElectricDifferentiator;
typedef KSMathIntegrator<KSTrajElectricSystem> KSTrajElectricIntegrator;
typedef KSMathInterpolator<KSTrajElectricSystem> KSTrajElectricInterpolator;

}  // namespace Kassiopeia

#endif
