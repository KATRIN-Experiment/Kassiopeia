#ifndef Kassiopeia_KSTrajMagneticTypes_h_
#define Kassiopeia_KSTrajMagneticTypes_h_

#include "KSMathSystem.h"
#include "KSMathControl.h"
#include "KSMathDifferentiator.h"
#include "KSMathIntegrator.h"
#include "KSMathInterpolator.h"

#include "KSTrajMagneticParticle.h"
#include "KSTrajMagneticDerivative.h"
#include "KSTrajMagneticError.h"

namespace Kassiopeia
{

    typedef KSMathSystem< KSTrajMagneticParticle, KSTrajMagneticDerivative, KSTrajMagneticError > KSTrajMagneticSystem;
    typedef KSMathControl< KSTrajMagneticSystem > KSTrajMagneticControl;
    typedef KSMathDifferentiator< KSTrajMagneticSystem > KSTrajMagneticDifferentiator;
    typedef KSMathIntegrator< KSTrajMagneticSystem > KSTrajMagneticIntegrator;
    typedef KSMathInterpolator< KSTrajMagneticSystem > KSTrajMagneticInterpolator;

}

#endif
