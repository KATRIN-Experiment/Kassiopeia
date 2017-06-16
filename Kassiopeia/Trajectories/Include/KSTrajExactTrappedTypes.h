#ifndef Kassiopeia_KSTrajExactTrappedTypes_h_
#define Kassiopeia_KSTrajExactTrappedTypes_h_

#include "KSMathSystem.h"
#include "KSMathControl.h"
#include "KSMathDifferentiator.h"
#include "KSMathIntegrator.h"
#include "KSMathInterpolator.h"

#include "KSTrajExactTrappedParticle.h"
#include "KSTrajExactTrappedDerivative.h"
#include "KSTrajExactTrappedError.h"

namespace Kassiopeia
{

    typedef KSMathSystem< KSTrajExactTrappedParticle, KSTrajExactTrappedDerivative, KSTrajExactTrappedError > KSTrajExactTrappedSystem;
    typedef KSMathControl< KSTrajExactTrappedSystem > KSTrajExactTrappedControl;
    typedef KSMathDifferentiator< KSTrajExactTrappedSystem > KSTrajExactTrappedDifferentiator;
    typedef KSMathIntegrator< KSTrajExactTrappedSystem > KSTrajExactTrappedIntegrator;
    typedef KSMathInterpolator< KSTrajExactTrappedSystem > KSTrajExactTrappedInterpolator;

}

#endif
