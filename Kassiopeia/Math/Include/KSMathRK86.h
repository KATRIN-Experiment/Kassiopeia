#ifndef Kassiopeia_KSMathRK86_h_
#define Kassiopeia_KSMathRK86_h_

#include "KSMathIntegrator.h"

namespace Kassiopeia
{

    template< class XSystemType >
    class KSMathRK86 :
        public KSMathIntegrator< XSystemType >
    {
        public:
            KSMathRK86();
            virtual ~KSMathRK86();

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
                eSubsteps = 12, eConditions = eSubsteps - 1
            };

            mutable ValueType fValues[ eConditions + 1 ];
            mutable DerivativeType fDerivatives[ eSubsteps ];

            static const double fA[ eSubsteps - 1 ][ eSubsteps - 1 ];
            static const double fAFinal6[ eSubsteps ];
            static const double fAFinal8[ eSubsteps ];
    };

    template< class XSystemType >
    KSMathRK86< XSystemType >::KSMathRK86()
    {
    }

    template< class XSystemType >
    KSMathRK86< XSystemType >::~KSMathRK86()
    {
    }

    template< class XSystemType >
    void KSMathRK86< XSystemType >::Integrate( const DifferentiatorType& aTerm, const ValueType& anInitialValue, const double& aStep, ValueType& aFinalValue, ErrorType& anError ) const
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

        fValues[ 7 ] = anInitialValue + aStep * (fA[ 7 ][ 0 ] * fDerivatives[ 0 ] + fA[ 7 ][ 1 ] * fDerivatives[ 1 ] + fA[ 7 ][ 2 ] * fDerivatives[ 2 ] + fA[ 7 ][ 3 ] * fDerivatives[ 3 ] + fA[ 7 ][ 4 ] * fDerivatives[ 4 ] + fA[ 7 ][ 5 ] * fDerivatives[ 5 ] + fA[ 7 ][ 6 ] * fDerivatives[ 6 ] + fA[ 7 ][ 7 ] * fDerivatives[ 7 ]);

        aTerm.Differentiate( fValues[ 7 ], fDerivatives[ 8 ] );

        fValues[ 8 ] = anInitialValue + aStep * (fA[ 8 ][ 0 ] * fDerivatives[ 0 ] + fA[ 8 ][ 1 ] * fDerivatives[ 1 ] + fA[ 8 ][ 2 ] * fDerivatives[ 2 ] + fA[ 8 ][ 3 ] * fDerivatives[ 3 ] + fA[ 8 ][ 4 ] * fDerivatives[ 4 ] + fA[ 8 ][ 5 ] * fDerivatives[ 5 ] + fA[ 8 ][ 6 ] * fDerivatives[ 6 ] + fA[ 8 ][ 7 ] * fDerivatives[ 7 ] + fA[ 8 ][ 8 ] * fDerivatives[ 8 ]);

        aTerm.Differentiate( fValues[ 8 ], fDerivatives[ 9 ] );

        fValues[ 9 ] = anInitialValue + aStep * (fA[ 9 ][ 0 ] * fDerivatives[ 0 ] + fA[ 9 ][ 1 ] * fDerivatives[ 1 ] + fA[ 9 ][ 2 ] * fDerivatives[ 2 ] + fA[ 9 ][ 3 ] * fDerivatives[ 3 ] + fA[ 9 ][ 4 ] * fDerivatives[ 4 ] + fA[ 9 ][ 5 ] * fDerivatives[ 5 ] + fA[ 9 ][ 6 ] * fDerivatives[ 6 ] + fA[ 9 ][ 7 ] * fDerivatives[ 7 ] + fA[ 9 ][ 8 ] * fDerivatives[ 8 ] + fA[ 9 ][ 9 ] * fDerivatives[ 9 ]);

        aTerm.Differentiate( fValues[ 9 ], fDerivatives[ 10 ] );

        fValues[ 10 ] = anInitialValue + aStep * (fA[ 10 ][ 0 ] * fDerivatives[ 0 ] + fA[ 10 ][ 1 ] * fDerivatives[ 1 ] + fA[ 10 ][ 2 ] * fDerivatives[ 2 ] + fA[ 10 ][ 3 ] * fDerivatives[ 3 ] + fA[ 10 ][ 4 ] * fDerivatives[ 4 ] + fA[ 10 ][ 5 ] * fDerivatives[ 5 ] + fA[ 10 ][ 6 ] * fDerivatives[ 6 ] + fA[ 10 ][ 7 ] * fDerivatives[ 7 ] + fA[ 10 ][ 8 ] * fDerivatives[ 8 ] + fA[ 10 ][ 9 ] * fDerivatives[ 9 ] + fA[ 10 ][ 10 ] * fDerivatives[ 10 ]);

        aTerm.Differentiate( fValues[ 10 ], fDerivatives[ 11 ] );

        fValues[ 11 ] = anInitialValue + aStep * (fAFinal6[ 0 ] * fDerivatives[ 0 ] + fAFinal6[ 1 ] * fDerivatives[ 1 ] + fAFinal6[ 2 ] * fDerivatives[ 2 ] + fAFinal6[ 3 ] * fDerivatives[ 3 ] + fAFinal6[ 4 ] * fDerivatives[ 4 ] + fAFinal6[ 5 ] * fDerivatives[ 5 ] + fAFinal6[ 6 ] * fDerivatives[ 6 ] + fAFinal6[ 7 ] * fDerivatives[ 7 ] + fAFinal6[ 8 ] * fDerivatives[ 8 ] + fAFinal6[ 9 ] * fDerivatives[ 9 ] + fAFinal6[ 10 ] * fDerivatives[ 10 ] + fAFinal6[ 11 ] * fDerivatives[ 11 ]);
        aFinalValue = anInitialValue + aStep * (fAFinal8[ 0 ] * fDerivatives[ 0 ] + fAFinal8[ 1 ] * fDerivatives[ 1 ] + fAFinal8[ 2 ] * fDerivatives[ 2 ] + fAFinal8[ 3 ] * fDerivatives[ 3 ] + fAFinal8[ 4 ] * fDerivatives[ 4 ] + fAFinal8[ 5 ] * fDerivatives[ 5 ] + fAFinal8[ 6 ] * fDerivatives[ 6 ] + fAFinal8[ 7 ] * fDerivatives[ 7 ] + fAFinal8[ 8 ] * fDerivatives[ 8 ] + fAFinal8[ 9 ] * fDerivatives[ 9 ] + fAFinal8[ 10 ] * fDerivatives[ 10 ] + fAFinal8[ 11 ] * fDerivatives[ 11 ]);
        anError = aStep * (aFinalValue - fValues[ 11 ]);

        return;
    }

    template< class XSystemType >
    const double KSMathRK86< XSystemType >::fA[ eSubsteps - 1 ][ eSubsteps - 1 ] =
    {
    { 9.0 / 142.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 }, //1st row
            { 178422123.0 / 9178574137.0, 685501333.0 / 8224473205.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 }, //2nd row
            { 12257.0 / 317988.0, 0.0, 12257.0 / 105996.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 }, //3rd row
            { 2584949729.0 / 6554704252.0, 0.0, -9163901916.0 / 6184003973.0, 26222057794.0 / 17776421907.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 }, //4th row
            { 4418011.0 / 96055225.0, 0.0, 0.0, 2947922107.0 / 12687381736.0, 3229973413.0 / 17234960414.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 }, //5th row
            { 2875139539.0 / 47877267651.0, 0.0, 0.0, 2702377211.0 / 24084535832.0, -135707089.0 / 4042230341.0, 299874140.0 / 17933325691.0, 0.0, 0.0, 0.0, 0.0, 0.0 }, //6th row
            { -7872176137.0 / 5003514694.0, 0.0, 0.0, -35136108789.0 / 26684798878.0, -114433184681.0 / 9760995895.0, 299204996517.0 / 32851421233.0, 254.0 / 39.0, 0.0, 0.0, 0.0, 0.0 }, //7th row
            { -3559950777.0 / 7399971898.0, 0.0, 0.0, -29299291531.0 / 4405504148.0, -42434013379.0 / 9366905709.0, 20642871700.0 / 5300635453.0, 12951197050.0 / 1499985011.0, 59527523.0 / 6331620793.0, 0.0, 0.0, 0.0 }, //8th row
            { -8196723582.0 / 10570795981.0, 0.0, 0.0, -46181454005.0 / 5775132776.0, -196277106011.0 / 29179424052.0, 63575135343.0 / 11491868333.0, 9348448139.0 / 857846776.0, 195434294.0 / 9727139945.0, -617468037.0 / 15757346105.0, 0.0, 0.0 }, //9th row
            { -6373809055.0 / 5357779452.0, 0.0, 0.0, -150772749657.0 / 21151088080.0, -58076657383.0 / 6089469394.0, 9252721190.0 / 1221566797.0, 132381309631.0 / 11748965576.0, 704633904.0 / 13813696331.0, 656417033.0 / 8185349658.0, -1669806516.0 / 10555289849.0, 0.0 }, //10th row
            { -2726346953.0 / 6954959789.0, 0.0, 0.0, 24906446731.0 / 6359105161.0, -65277767625.0 / 23298960463.0, 39128152317.0 / 16028215273.0, -40638357893.0 / 16804059016.0, -7437361171.0 / 21911114743.0, 1040125706.0 / 5334949109.0, -1129865134.0 / 5812907645.0, 6253441118.0 / 10543852725.0 } //11th row
    };

    template< class XSystemType >
    const double KSMathRK86< XSystemType >::fAFinal6[ eSubsteps ] =
    { 289283091.0 / 6008696510.0, 0.0, 0.0, 0.0, 0.0, 3034152487.0 / 7913336319.0, 7170564158.0 / 30263027435.0, 7206303747.0 / 16758195910.0, -1059739258.0 / 8472387467.0, 16534129531.0 / 11550853505.0, -3.0 / 2.0, 5118195927.0 / 53798651926.0 };

    template< class XSystemType >
    const double KSMathRK86< XSystemType >::fAFinal8[ eSubsteps ] =
    { 438853193.0 / 9881496838.0, 0.0, 0.0, 0.0, 0.0, 11093525429.0 / 31342013414.0, 481311443.0 / 1936695762.0, -3375294558.0 / 10145424253.0, 9830993862.0 / 5116981057.0, -138630849943.0 / 50747474617.0, 7152278206.0 / 5104393345.0, 5118195927.0 / 53798651926.0 };

}

#endif
