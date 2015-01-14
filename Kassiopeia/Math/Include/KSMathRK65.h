#ifndef Kassiopeia_KSMathRK65_h_
#define Kassiopeia_KSMathRK65_h_

#include "KSMathIntegrator.h"

namespace Kassiopeia
{

    template< class XSystemType >
    class KSMathRK65 :
        public KSMathIntegrator< XSystemType >
    {
        public:
            KSMathRK65();
            virtual ~KSMathRK65();

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
                eSubsteps = 8, eConditions = eSubsteps - 1
            };

            mutable ValueType fValues[ eConditions + 1 ];
            mutable DerivativeType fDerivatives[ eSubsteps ];

            static const double fA[ eSubsteps - 1 ][ eSubsteps - 1 ];
            static const double fAFinal5[ eSubsteps ];
            static const double fAFinal6[ eSubsteps ];
    };

    template< class XSystemType >
    KSMathRK65< XSystemType >::KSMathRK65()
    {
    }

    template< class XSystemType >
    KSMathRK65< XSystemType >::~KSMathRK65()
    {
    }

    template< class XSystemType >
    void KSMathRK65< XSystemType >::Integrate( const DifferentiatorType& aTerm, const ValueType& anInitialValue, const double& aStep, ValueType& aFinalValue, ErrorType& anError ) const
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

        fValues[ 5 ] = anInitialValue + aStep * (fA[ 5 ][ 0 ] * fDerivatives[ 0 ] + fA[ 5 ][ 1 ] * fDerivatives[ 1 ] + fA[ 5 ][ 2 ] * fDerivatives[ 2 ] + fA[ 5 ][ 3 ] * fDerivatives[ 3 ] + fA[ 5 ][ 4 ] * fDerivatives[ 4 ] + fA[ 5 ][ 5 ] * fDerivatives[ 5 ]);

        aTerm.Differentiate( fValues[ 5 ], fDerivatives[ 6 ] );

        fValues[ 6 ] = anInitialValue + aStep * (fA[ 6 ][ 0 ] * fDerivatives[ 0 ] + fA[ 6 ][ 1 ] * fDerivatives[ 1 ] + fA[ 6 ][ 2 ] * fDerivatives[ 2 ] + fA[ 6 ][ 3 ] * fDerivatives[ 3 ] + fA[ 6 ][ 4 ] * fDerivatives[ 4 ] + fA[ 6 ][ 5 ] * fDerivatives[ 5 ] + fA[ 6 ][ 6 ] * fDerivatives[ 6 ]);

        aTerm.Differentiate( fValues[ 6 ], fDerivatives[ 7 ] );

        fValues[ 7 ] = anInitialValue + aStep * (fAFinal5[ 0 ] * fDerivatives[ 0 ] + fAFinal5[ 1 ] * fDerivatives[ 1 ] + fAFinal5[ 2 ] * fDerivatives[ 2 ] + fAFinal5[ 3 ] * fDerivatives[ 3 ] + fAFinal5[ 4 ] * fDerivatives[ 4 ] + fAFinal5[ 5 ] * fDerivatives[ 5 ] + fAFinal5[ 6 ] * fDerivatives[ 6 ] + fAFinal5[ 7 ] * fDerivatives[ 7 ]);
        aFinalValue = anInitialValue + aStep * (fAFinal6[ 0 ] * fDerivatives[ 0 ] + fAFinal6[ 1 ] * fDerivatives[ 1 ] + fAFinal6[ 2 ] * fDerivatives[ 2 ] + fAFinal6[ 3 ] * fDerivatives[ 3 ] + fAFinal6[ 4 ] * fDerivatives[ 4 ] + fAFinal6[ 5 ] * fDerivatives[ 5 ] + fAFinal6[ 6 ] * fDerivatives[ 6 ] + fAFinal6[ 7 ] * fDerivatives[ 7 ]);
        anError = aStep * (aFinalValue - fValues[ 7 ]);

        return;
    }

    template< class XSystemType >
    const double KSMathRK65< XSystemType >::fA[ eSubsteps - 1 ][ eSubsteps - 1 ] =
    {
    { 1. / 10., 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 },
    { -2. / 81., 20. / 81., 0.0, 0.0, 0.0, 0.0, 0.0 },
    { 615. / 1372., -270. / 343., 1053. / 1372., 0., 0., 0., 0. },
    { 3243. / 5500., -54. / 55., 50949. / 71500., 4998. / 17875., 0., 0., 0. },
    { -26492. / 37125., 72. / 55., 2808. / 23375., -24206. / 37125., 338. / 459., 0., 0. },
    { 5561. / 2376., -35. / 11., -24117. / 31603., 899983. / 200772., -5225. / 1836., 3925. / 4056., 0. },
    { 465467. / 266112., -2945. / 1232., -5610201. / 14158144., 10513573. / 3212352., -424325. / 205632., 376225. / 454272., 0. } };

    template< class XSystemType >
    const double KSMathRK65< XSystemType >::fAFinal5[ eSubsteps ] =
    { 821. / 10800., 0., 19683. / 71825., 175273. / 912600., 395. / 3672., 785. / 2704., 3. / 50., 0. };

    template< class XSystemType >
    const double KSMathRK65< XSystemType >::fAFinal6[ eSubsteps ] =
    { 61. / 864., 0., 98415. / 321776., 16807. / 146016., 1375. / 7344., 1375. / 5408., -37. / 1120., 1. / 10. };

}

#endif
