#ifndef Kassiopeia_KSIntDecayCalculatorDeathConstRate_h_
#define Kassiopeia_KSIntDecayCalculatorDeathConstRate_h_

#include "KSIntDecayCalculator.h"
#include "KField.h"

namespace Kassiopeia
{
    class KSIntDecayCalculatorDeathConstRate :
        public KSComponentTemplate< KSIntDecayCalculatorDeathConstRate, KSIntDecayCalculator >
    {
        public:
            KSIntDecayCalculatorDeathConstRate();
            KSIntDecayCalculatorDeathConstRate( const KSIntDecayCalculatorDeathConstRate& aCopy );
            KSIntDecayCalculatorDeathConstRate* Clone() const;
            virtual ~KSIntDecayCalculatorDeathConstRate();

        public:
            void CalculateLifeTime( const KSParticle& aParticle, double& aLifeTime );
            void ExecuteInteraction( const KSParticle& anInitialParticle,
                                     KSParticle& aFinalParticle,
                                     KSParticleQueue& aSecondaries );


        public:
            K_SET_GET( double, LifeTime ) // s
            K_SET_GET( long long, TargetPID )
            K_SET_GET( long long, minPID )
            K_SET_GET( long long, maxPID )

            public:
                void SetDecayProductGenerator( KSGenerator* const aGenerator );
                KSGenerator* GetDecayProductGenerator() const;

            protected:
                KSGenerator* fDecayProductGenerator;
    };


}

#endif
