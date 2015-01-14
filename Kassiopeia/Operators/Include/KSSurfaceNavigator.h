#ifndef Kassiopeia_KSSurfaceNavigator_h_
#define Kassiopeia_KSSurfaceNavigator_h_

#include "KSComponentTemplate.h"
#include "KSParticle.h"

namespace Kassiopeia
{

    class KSSurfaceNavigator :
        public KSComponentTemplate< KSSurfaceNavigator >
    {
        public:
            KSSurfaceNavigator();
            virtual ~KSSurfaceNavigator();

        public:
            virtual void ExecuteNavigation(
                const KSParticle& anInitialParticle,
                KSParticle& aFinalParticle,
                KSParticleQueue& aSecondaries
            ) const = 0;
    };

}

#endif
