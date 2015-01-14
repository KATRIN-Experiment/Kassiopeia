#ifndef Kassiopeia_KSMathRK54_h_
#define Kassiopeia_KSMathRK54_h_

#include "KSMathIntegrator.h"

namespace Kassiopeia
{

    template< class XSystemType >
    class KSMathRK54 :
        public KSMathIntegrator< XSystemType >
    {
        public:
            KSMathRK54();
            virtual ~KSMathRK54();

        public:
            typedef XSystemType SystemType;
            typedef KSMathDifferentiator< SystemType > DifferentiatorType;
            typedef typename SystemType::ValueType ValueType;
            typedef typename SystemType::DerivativeType DerivativeType;
            typedef typename SystemType::ErrorType ErrorType;

        public:
            virtual void Integrate( const DifferentiatorType& aTerm, const ValueType& anInitialValue, const double& aStep, ValueType& aFinalValue, ErrorType& anError ) const;

        private:
            enum
            {
                eSubsteps = 6, eConditions = eSubsteps - 1
            };

            mutable ValueType fValues[ eConditions + 1 ];
            mutable DerivativeType fDerivatives[ eSubsteps ];

            static const double fA[ eSubsteps - 1 ][ eSubsteps - 1 ];
            static const double fAFinal4[ eSubsteps ];
            static const double fAFinal5[ eSubsteps ];
    };

    template< class XSystemType >
    KSMathRK54< XSystemType >::KSMathRK54()
    {
    }

    template< class XSystemType >
    KSMathRK54< XSystemType >::~KSMathRK54()
    {
    }

    template< class XSystemType >
    void KSMathRK54< XSystemType >::Integrate( const DifferentiatorType& aTerm, const ValueType& anInitialValue, const double& aStep, ValueType& aFinalValue, ErrorType& anError ) const
    {
        aTerm.Differentiate( anInitialValue, fDerivatives[ 0 ] );

        fValues[ 0 ] = anInitialValue + aStep * (fA[ 0 ][ 0 ] * fDerivatives[ 0 ]);

        aTerm.Differentiate( fValues[ 0 ], fDerivatives[ 1 ] );

        fValues[ 1 ] = anInitialValue + aStep * (fA[ 1 ][ 0 ] * fDerivatives[ 0 ] + fA[ 1 ][ 1 ] * fDerivatives[ 1 ]);

        aTerm.Differentiate( fValues[ 1 ], fDerivatives[ 2 ] );

        fValues[ 2 ] = anInitialValue + aStep * (fA[ 2 ][ 0 ] * fDerivatives[ 0 ] + fA[ 2 ][ 1 ] * fDerivatives[ 1 ] + fA[ 2 ][ 2 ] * fDerivatives[ 2 ]);

        aTerm.Differentiate( fValues[ 2 ], fDerivatives[ 3 ] );

        fValues[ 3 ] = anInitialValue + aStep * (fA[ 3 ][ 0 ] * fDerivatives[ 0 ] + fA[ 3 ][ 1 ] * fDerivatives[ 1 ] + fA[ 3 ][ 2 ] * fDerivatives[ 2 ] + fA[ 3 ][ 3 ] * fDerivatives[ 3 ]);

        aTerm.Differentiate( fValues[ 3 ], fDerivatives[ 4 ] );

        fValues[ 4 ] = anInitialValue + aStep * (fA[ 4 ][ 0 ] * fDerivatives[ 0 ] + fA[ 4 ][ 1 ] * fDerivatives[ 1 ] + fA[ 4 ][ 2 ] * fDerivatives[ 2 ] + fA[ 4 ][ 3 ] * fDerivatives[ 3 ] + fA[ 4 ][ 4 ] * fDerivatives[ 4 ]);

        aTerm.Differentiate( fValues[ 4 ], fDerivatives[ 5 ] );

        fValues[ 5 ] = anInitialValue + aStep * (fAFinal4[ 0 ] * fDerivatives[ 0 ] + fAFinal4[ 1 ] * fDerivatives[ 1 ] + fAFinal4[ 2 ] * fDerivatives[ 2 ] + fAFinal4[ 3 ] * fDerivatives[ 3 ] + fAFinal4[ 4 ] * fDerivatives[ 4 ] + fAFinal4[ 5 ] * fDerivatives[ 5 ]);
        aFinalValue = anInitialValue + aStep * (fAFinal5[ 0 ] * fDerivatives[ 0 ] + fAFinal5[ 1 ] * fDerivatives[ 1 ] + fAFinal5[ 2 ] * fDerivatives[ 2 ] + fAFinal5[ 3 ] * fDerivatives[ 3 ] + fAFinal5[ 4 ] * fDerivatives[ 4 ] + fAFinal5[ 5 ] * fDerivatives[ 5 ]);
        anError = aStep * (aFinalValue - fValues[ 5 ]);

        return;
    }

    template< class XSystemType >
    const double KSMathRK54< XSystemType >::fA[ eSubsteps - 1 ][ eSubsteps - 1 ] =
    {
    { 1.0 / 4.0, 0.0, 0.0, 0.0, 0.0 },
    { 3.0 / 32.0, 9.0 / 32.0, 0.0, 0.0, 0.0 },
    { 1932.0 / 2197.0, -7200.0 / 2197.0, 7296.0 / 2197.0, 0.0, 0.0 },
    { 439.0 / 216.0, -8.0, 3680.0 / 513.0, -845.0 / 4104.0, 0.0 },
    { -8.0 / 27.0, 2.0, -3544.0 / 2565.0, 1859.0 / 4104.0, -11.0 / 40.0 } };

    template< class XSystemType >
    const double KSMathRK54< XSystemType >::fAFinal4[ eSubsteps ] =
    { 25.0 / 216.0, 0.0, 1408.0 / 2565.0, 2197.0 / 4104.0, -1.0 / 5.0, 0.0 };

    template< class XSystemType >
    const double KSMathRK54< XSystemType >::fAFinal5[ eSubsteps ] =
    { 16.0 / 135.0, 0.0, 6656.0 / 12825.0, 28561.0 / 56430.0, -9.0 / 50.0, 2.0 / 55.0 };

}

#endif

