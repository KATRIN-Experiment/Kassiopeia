#ifndef Kassiopeia_KSMathIntegrator_h_
#define Kassiopeia_KSMathIntegrator_h_

#include "KSMathSystem.h"
#include "KSMathDifferentiator.h"

namespace Kassiopeia
{
    template< class XType >
    class KSMathIntegrator;

    template< class XValueType, class XDerivativeType, class XErrorType >
    class KSMathIntegrator< KSMathSystem< XValueType, XDerivativeType, XErrorType > >
    {
        public:
            KSMathIntegrator();
            virtual ~KSMathIntegrator();

        public:
            virtual void Integrate( const KSMathDifferentiator< KSMathSystem< XValueType, XDerivativeType, XErrorType > >& aTerm, const XValueType& anInitialValue, const double& aStep, XValueType& aFinalValue, XErrorType& anError ) const = 0;
    };

    template< class XValueType, class XDerivativeType, class XErrorType >
    KSMathIntegrator< KSMathSystem< XValueType, XDerivativeType, XErrorType > >::KSMathIntegrator()
    {
    }

    template< class XValueType, class XDerivativeType, class XErrorType >
    KSMathIntegrator< KSMathSystem< XValueType, XDerivativeType, XErrorType > >::~KSMathIntegrator()
    {
    }

}

#endif
