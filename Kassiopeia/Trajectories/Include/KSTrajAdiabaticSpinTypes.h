#ifndef Kassiopeia_KSTrajAdiabaticSpinTypes_h_
#define Kassiopeia_KSTrajAdiabaticSpinTypes_h_

#include "KSMathSystem.h"
#include "KSMathControl.h"
#include "KSMathDifferentiator.h"
#include "KSMathIntegrator.h"
#include "KSMathInterpolator.h"

#include "KSTrajAdiabaticSpinParticle.h"
#include "KSTrajAdiabaticSpinDerivative.h"
#include "KSTrajAdiabaticSpinError.h"

namespace Kassiopeia
{

    typedef KSMathSystem< KSTrajAdiabaticSpinParticle, KSTrajAdiabaticSpinDerivative, KSTrajAdiabaticSpinError > KSTrajAdiabaticSpinSystem;
    typedef KSMathControl< KSTrajAdiabaticSpinSystem > KSTrajAdiabaticSpinControl;
    typedef KSMathDifferentiator< KSTrajAdiabaticSpinSystem > KSTrajAdiabaticSpinDifferentiator;
    typedef KSMathIntegrator< KSTrajAdiabaticSpinSystem > KSTrajAdiabaticSpinIntegrator;
    typedef KSMathInterpolator< KSTrajAdiabaticSpinSystem > KSTrajAdiabaticSpinInterpolator;

}

#endif
