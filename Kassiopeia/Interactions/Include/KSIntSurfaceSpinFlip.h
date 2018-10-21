#ifndef Kassiopeia_KSIntSurfaceSpinFlip_h_
#define Kassiopeia_KSIntSurfaceSpinFlip_h_

#include "KSSurfaceInteraction.h"

#include "KField.h"

namespace Kassiopeia
{

    class KSStep;

    class KSIntSurfaceSpinFlip :
        public KSComponentTemplate< KSIntSurfaceSpinFlip, KSSurfaceInteraction >
    {
        public:
            KSIntSurfaceSpinFlip();
            KSIntSurfaceSpinFlip( const KSIntSurfaceSpinFlip& aCopy );
            KSIntSurfaceSpinFlip* Clone() const;
            virtual ~KSIntSurfaceSpinFlip();

        public:
            void ExecuteInteraction( const KSParticle& anInitialParticle, KSParticle& aFinalParticle, KSParticleQueue& aSecondaries );

        public:

            K_SET_GET( double, Probability )

    };

}

#endif
