#ifndef Kassiopeia_KSMathSystem_h_
#define Kassiopeia_KSMathSystem_h_

#include "KAssert.h"

#include <cstddef>

namespace Kassiopeia
{

    template< class XValueType, class XDerivativeType, class XErrorType >
    class KSMathSystem
    {
        public:
            typedef XValueType ValueType;
            typedef XDerivativeType DerivativeType;
            typedef XErrorType ErrorType;

        public:
            KSMathSystem();
            virtual ~KSMathSystem();
    };

    template< class XValueType, class XDerivativeType, class XErrorType >
    KSMathSystem< XValueType, XDerivativeType, XErrorType >::KSMathSystem()
    {
        KSTATICASSERT( ValueType::eDimension == DerivativeType::eDimension, dimension_mismatch_between_value_and_derivative_types_in_KMathSystem )
        KSTATICASSERT( ValueType::eDimension == ErrorType::eDimension, dimension_mismatch_between_value_and_error_types_in_KMathSystem )
    }

    template< class XValueType, class XDerivativeType, class XErrorType >
    KSMathSystem< XValueType, XDerivativeType, XErrorType >::~KSMathSystem()
    {
    }

}

#endif
