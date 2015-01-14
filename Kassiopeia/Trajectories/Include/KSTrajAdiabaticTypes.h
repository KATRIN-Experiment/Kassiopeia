#ifndef Kassiopeia_KSTrajAdiabaticTypes_h_
#define Kassiopeia_KSTrajAdiabaticTypes_h_

#include "KSMathSystem.h"
#include "KSMathControl.h"
#include "KSMathDifferentiator.h"
#include "KSMathIntegrator.h"
#include "KSMathInterpolator.h"

#include "KSTrajAdiabaticParticle.h"
#include "KSTrajAdiabaticDerivative.h"
#include "KSTrajAdiabaticError.h"

namespace Kassiopeia
{

    typedef KSMathSystem< KSTrajAdiabaticParticle, KSTrajAdiabaticDerivative, KSTrajAdiabaticError > KSTrajAdiabaticSystem;
    typedef KSMathControl< KSTrajAdiabaticSystem > KSTrajAdiabaticControl;
    typedef KSMathDifferentiator< KSTrajAdiabaticSystem > KSTrajAdiabaticDifferentiator;
    typedef KSMathIntegrator< KSTrajAdiabaticSystem > KSTrajAdiabaticIntegrator;
    typedef KSMathInterpolator< KSTrajAdiabaticSystem > KSTrajAdiabaticInterpolator;

}

#endif
