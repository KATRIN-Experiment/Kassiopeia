#ifndef Kassiopeia_KSMathRK87_h_
#define Kassiopeia_KSMathRK87_h_

#include "KSMathIntegrator.h"

namespace Kassiopeia
{

    template< class XSystemType >
    class KSMathRK87 :
        public KSMathIntegrator< XSystemType >
    {
        public:
            KSMathRK87();
            virtual ~KSMathRK87();

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
                eSubsteps = 13, eConditions = eSubsteps - 1
            };

            mutable ValueType fValues[ eConditions + 1 ];
            mutable DerivativeType fDerivatives[ eSubsteps ];

            static const double fA[ eSubsteps - 1 ][ eSubsteps - 1 ];
            static const double fAFinal7[ eSubsteps ];
            static const double fAFinal8[ eSubsteps ];
    };

    template< class XSystemType >
    KSMathRK87< XSystemType >::KSMathRK87()
    {
    }

    template< class XSystemType >
    KSMathRK87< XSystemType >::~KSMathRK87()
    {
    }

    template< class XSystemType >
    void KSMathRK87< XSystemType >::Integrate( const DifferentiatorType& aTerm, const ValueType& anInitialValue, const double& aStep, ValueType& aFinalValue, ErrorType& anError ) const
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

        fValues[ 11 ] = anInitialValue + aStep * (fA[ 11 ][ 0 ] * fDerivatives[ 0 ] + fA[ 11 ][ 1 ] * fDerivatives[ 1 ] + fA[ 11 ][ 2 ] * fDerivatives[ 2 ] + fA[ 11 ][ 3 ] * fDerivatives[ 3 ] + fA[ 11 ][ 4 ] * fDerivatives[ 4 ] + fA[ 11 ][ 5 ] * fDerivatives[ 5 ] + fA[ 11 ][ 6 ] * fDerivatives[ 6 ] + fA[ 11 ][ 7 ] * fDerivatives[ 7 ] + fA[ 11 ][ 8 ] * fDerivatives[ 8 ] + fA[ 11 ][ 9 ] * fDerivatives[ 9 ] + fA[ 11 ][ 10 ] * fDerivatives[ 10 ] + fA[ 11 ][ 11 ] * fDerivatives[ 11 ]);

        aTerm.Differentiate( fValues[ 11 ], fDerivatives[ 12 ] );

        fValues[ 12 ] = anInitialValue + aStep * (fAFinal7[ 0 ] * fDerivatives[ 0 ] + fAFinal7[ 1 ] * fDerivatives[ 1 ] + fAFinal7[ 2 ] * fDerivatives[ 2 ] + fAFinal7[ 3 ] * fDerivatives[ 3 ] + fAFinal7[ 4 ] * fDerivatives[ 4 ] + fAFinal7[ 5 ] * fDerivatives[ 5 ] + fAFinal7[ 6 ] * fDerivatives[ 6 ] + fAFinal7[ 7 ] * fDerivatives[ 7 ] + fAFinal7[ 8 ] * fDerivatives[ 8 ] + fAFinal7[ 9 ] * fDerivatives[ 9 ] + fAFinal7[ 10 ] * fDerivatives[ 10 ] + fAFinal7[ 11 ] * fDerivatives[ 11 ] + fAFinal7[ 12 ] * fDerivatives[ 12 ]);
        aFinalValue = anInitialValue + aStep * (fAFinal8[ 0 ] * fDerivatives[ 0 ] + fAFinal8[ 1 ] * fDerivatives[ 1 ] + fAFinal8[ 2 ] * fDerivatives[ 2 ] + fAFinal8[ 3 ] * fDerivatives[ 3 ] + fAFinal8[ 4 ] * fDerivatives[ 4 ] + fAFinal8[ 5 ] * fDerivatives[ 5 ] + fAFinal8[ 6 ] * fDerivatives[ 6 ] + fAFinal8[ 7 ] * fDerivatives[ 7 ] + fAFinal8[ 8 ] * fDerivatives[ 8 ] + fAFinal8[ 9 ] * fDerivatives[ 9 ] + fAFinal8[ 10 ] * fDerivatives[ 10 ] + fAFinal8[ 11 ] * fDerivatives[ 11 ] + fAFinal8[ 12 ] * fDerivatives[ 12 ]);
        anError = aStep * (aFinalValue - fValues[ 12 ]);

        return;
    }

    template< class XSystemType >
    const double KSMathRK87< XSystemType >::fA[ eSubsteps - 1 ][ eSubsteps - 1 ] =
    {
    { 1. / 18., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0. }, //1st row
            { 1. / 48., 1. / 16., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0. }, //2nd row
            { 1. / 32., 0., 3. / 32., 0., 0., 0., 0., 0., 0., 0., 0., 0. }, //3rd row
            { 5. / 16., 0., -75. / 64., 75. / 64., 0., 0., 0., 0., 0., 0., 0., 0. }, //4th row
            { 3. / 80., 0., 0., 3. / 16., 3. / 20., 0., 0., 0., 0., 0., 0., 0. }, //5th row
            { 29443841. / 614563906., 0., 0., 77736538. / 692538347., -28693883. / 1125000000., 23124283. / 1800000000., 0., 0., 0., 0., 0., 0. }, //6th row
            { 16016141. / 946692911., 0., 0., 61564180. / 158732637., 22789713. / 633445777., 545815736. / 2771057229., -180193667. / 1043307555., 0., 0., 0., 0., 0. }, //7th row
            { 39632708. / 573591083., 0., 0., -433636366. / 683701615., -421739975. / 2616292301., 100302831. / 723423059., 790204164. / 839813087., 800635310. / 3783071287., 0., 0., 0., 0. }, //8th row
            { 246121993. / 1340847787., 0., 0., -37695042795. / 15268766246., -309121744. / 1061227803., -12992083. / 490766935., 6005943493. / 2108947869., 393006217. / 1396673457., 123872331. / 1001029789., 0., 0., 0. }, //9th row
            { -1028468189. / 846180014., 0., 0., 8478235783. / 508512852., 1311729495. / 1432422823., -10304129995. / 1701304382., -48777925059. / 3047939560., 15336726248. / 1032824649., -45442868181. / 3398467696., 3065993473. / 597172653., 0., 0. }, //10th row
            { 185892177. / 718116043., 0., 0., -3185094517. / 667107341., -477755414. / 1098053517., -703635378. / 230739211., 5731566787. / 1027545527., 5232866602. / 850066563., -4093664535. / 808688257., 3962137247. / 1805957418., 65686358. / 487910083., 0. }, //11th row
            { 403863854. / 491063109., 0., 0., -5068492393. / 434740067., -411421997. / 543043805., 652783627. / 914296604., 11173962825. / 925320556., -13158990841. / 6184727034., 3936647629. / 1978049680., -160528059. / 685178525., 248638103. / 1413531060., 0. } //12th row

    };

    template< class XSystemType >
    const double KSMathRK87< XSystemType >::fAFinal7[ eSubsteps ] =
    { 13451932. / 455176623., 0., 0., 0., 0., -808719846. / 976000145., 1757004468. / 5645159321., 656045339. / 265891186., -3867574721. / 1518517206., 465885868. / 322736535., 53011238. / 667516719., 2. / 45., 0. };

    template< class XSystemType >
    const double KSMathRK87< XSystemType >::fAFinal8[ eSubsteps ] =
    { 14005451. / 335480064., 0., 0., 0., 0., -59238493. / 1068277825., 181606767. / 758867731., 561292985. / 797845732., -1041891430. / 1371343529., 760417239. / 1151165299., 118820643. / 751138087., -528747749. / 2220607170., 1. / 4. };

}

#endif
