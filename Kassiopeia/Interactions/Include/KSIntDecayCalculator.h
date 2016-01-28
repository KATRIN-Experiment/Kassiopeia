#ifndef Kassiopeia_KSIntDecayCalculator_h_
#define Kassiopeia_KSIntDecayCalculator_h_

#include "KSComponentTemplate.h"
#include "KSGenerator.h"
#include "KSParticle.h"

#include "KField.h"

namespace Kassiopeia
{

    class KSIntDecayCalculator :
        public KSComponentTemplate< KSIntDecayCalculator >
    {
        public:
            KSIntDecayCalculator();
            virtual ~KSIntDecayCalculator();
            virtual KSIntDecayCalculator* Clone() const = 0;

        public:
            virtual void CalculateLifeTime( const KSParticle& aParticle, double& aCrossSection ) = 0;
            virtual void ExecuteInteraction( const KSParticle& anInitialParticle,
                                             KSParticle& aFinalParticle,
                                             KSParticleQueue& aSecondaries ) = 0;

        protected:
            virtual void PullDeupdateComponent();
            virtual void PushDeupdateComponent();


            //variables for output
            K_REFS( int, StepNDecays )
            K_REFS( double, StepEnergyLoss )
    };

}

#endif
