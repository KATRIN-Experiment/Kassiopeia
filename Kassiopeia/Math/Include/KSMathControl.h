#ifndef Kassiopeia_KSMathControl_h_
#define Kassiopeia_KSMathControl_h_

#include "KSMathSystem.h"

namespace Kassiopeia
{

    template< class XSystemType >
    class KSMathControl;

    template< class XValueType, class XDerivativeType, class XErrorType >
    class KSMathControl< KSMathSystem< XValueType, XDerivativeType, XErrorType > >
    {
        public:
            KSMathControl();
            virtual ~KSMathControl();

        public:
            virtual void Calculate( const XValueType& aCurrentValue, double& anIncrement ) = 0;
            virtual void Check( const XValueType& anInitialValue, const XValueType& aFinalValue, const XErrorType& anError, bool& aFlag ) = 0;
    };

    template< class XValueType, class XDerivativeType, class XErrorType >
    KSMathControl< KSMathSystem< XValueType, XDerivativeType, XErrorType > >::KSMathControl()
    {
    }

    template< class XValueType, class XDerivativeType, class XErrorType >
    KSMathControl< KSMathSystem< XValueType, XDerivativeType, XErrorType > >::~KSMathControl()
    {
    }

}

#endif
