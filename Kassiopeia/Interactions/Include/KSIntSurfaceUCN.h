#ifndef Kassiopeia_KSIntSurfaceUCN_h_
#define Kassiopeia_KSIntSurfaceUCN_h_

#include "KSSurfaceInteraction.h"

#include "KField.h"

namespace Kassiopeia
{

    class KSStep;

    class KSIntSurfaceUCN :
        public KSComponentTemplate< KSIntSurfaceUCN, KSSurfaceInteraction >
    {
        public:
            KSIntSurfaceUCN();
            KSIntSurfaceUCN( const KSIntSurfaceUCN& aCopy );
            KSIntSurfaceUCN* Clone() const;
            virtual ~KSIntSurfaceUCN();

        public:
            void ExecuteInteraction( const KSParticle& anInitialParticle, KSParticle& aFinalParticle, KSParticleQueue& aSecondaries );
            void ExecuteReflection( const KSParticle& anInitialParticle, KSParticle& aFinalParticle, KSParticleQueue& aSecondaries );
            void ExecuteTransmission( const KSParticle& anInitialParticle, KSParticle& aFinalParticle, KSParticleQueue& aSecondaries );

        public:
            K_SET_GET( double, Probability ) // transmission probability
            K_SET_GET( double, SpinFlipProbability)

    };

}

#endif
