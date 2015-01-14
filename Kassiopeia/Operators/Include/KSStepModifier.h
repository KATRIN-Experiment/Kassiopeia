#ifndef Kassiopeia_KSStepModifier_h_
#define Kassiopeia_KSStepModifier_h_

#include "KSComponentTemplate.h"
#include "KSParticle.h"

namespace Kassiopeia
{

    class KSTrack;

    class KSStepModifier:
            public KSComponentTemplate< KSStepModifier >
    {
        public:
            KSStepModifier();
            virtual ~KSStepModifier();

        public:
            virtual void ExecutePreStepModification( KSParticle& anInitialParticle,
                                                     KSParticleQueue& aQueue ) = 0;
            virtual void ExecutePostStepModifcation( KSParticle& anInitialParticle,
                                                     KSParticle& aFinalParticle,
                                                     KSParticleQueue& aQueue ) = 0;
    };

}

#endif
