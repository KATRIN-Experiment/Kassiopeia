#ifndef Kassiopeia_KSMathInterpolator_h_
#define Kassiopeia_KSMathInterpolator_h_

#include "KSMathSystem.h"
#include "KSMathDifferentiator.h"

namespace Kassiopeia
{
    template< class XType >
    class KSMathInterpolator;

    template< class XValueType, class XDerivativeType, class XErrorType >
    class KSMathInterpolator< KSMathSystem< XValueType, XDerivativeType, XErrorType > >
    {
        public:
            KSMathInterpolator();
            virtual ~KSMathInterpolator();

        public:
            virtual void Interpolate( const KSMathDifferentiator< KSMathSystem< XValueType, XDerivativeType, XErrorType > >& aDifferentiator, const XValueType& anInitialValue, const XValueType& aFinalValue, const double& aStep, XValueType& anInterpolatedValue ) const = 0;
    };

    template< class XValueType, class XDerivativeType, class XErrorType >
    KSMathInterpolator< KSMathSystem< XValueType, XDerivativeType, XErrorType > >::KSMathInterpolator()
    {
    }

    template< class XValueType, class XDerivativeType, class XErrorType >
    KSMathInterpolator< KSMathSystem< XValueType, XDerivativeType, XErrorType > >::~KSMathInterpolator()
    {
    }

}

#endif
