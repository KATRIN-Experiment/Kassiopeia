//
// Created by trost on 27.05.15.
//

#ifndef KASPER_KSINTDECAYCALCULATORFERENCBBRTRANSITION_H
#define KASPER_KSINTDECAYCALCULATORFERENCBBRTRANSITION_H

#include "KSIntDecayCalculator.h"
#include "RydbergFerenc.h"
#include "KField.h"

namespace Kassiopeia
{
    class KSIntDecayCalculatorFerencBBRTransition :
            public KSComponentTemplate< KSIntDecayCalculatorFerencBBRTransition, KSIntDecayCalculator >
    {
    public:
        KSIntDecayCalculatorFerencBBRTransition();
        KSIntDecayCalculatorFerencBBRTransition( const KSIntDecayCalculatorFerencBBRTransition& aCopy );
        KSIntDecayCalculatorFerencBBRTransition* Clone() const;
        virtual ~KSIntDecayCalculatorFerencBBRTransition();

    public:
        void CalculateLifeTime( const KSParticle& aParticle, double& aLifeTime );
        void ExecuteInteraction( const KSParticle& anInitialParticle,
                                 KSParticle& aFinalParticle,
                                 KSParticleQueue& aSecondaries );


    public:
        K_SET_GET( long long, TargetPID )
        K_SET_GET( long long, minPID )
        K_SET_GET( long long, maxPID )
        K_SET_GET( double, Temperature )

    protected:
        virtual void InitializeComponent();

    private:
        int fLastn;
        int fLastl;
        double fLastLifetime;
        double low_n_lifetimes[150][150];

    private:
        RydbergCalculator* fCalculator;

    };


}


#endif //KASPER_KSINTDECAYCALCULATORFERENCBBRTRANSITION_H
