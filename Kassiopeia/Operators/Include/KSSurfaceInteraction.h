#ifndef Kassiopeia_KSSurfaceInteraction_h_
#define Kassiopeia_KSSurfaceInteraction_h_

#include "KSComponentTemplate.h"
#include "KSParticle.h"

namespace Kassiopeia
{

    class KSSurfaceInteraction :
        public KSComponentTemplate< KSSurfaceInteraction >
    {
        public:
        	KSSurfaceInteraction();
            virtual ~KSSurfaceInteraction();

        public:
            virtual void ExecuteInteraction(
                const KSParticle& anInitialParticle,
                KSParticle& aFinalParticle,
                KSParticleQueue& aSecondaries
            ) = 0;
    };

}

#endif
