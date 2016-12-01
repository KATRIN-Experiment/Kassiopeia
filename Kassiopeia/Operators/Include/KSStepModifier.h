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

            //returns true if any of the state variables of anInitialParticle are changed
            virtual bool ExecutePreStepModification( KSParticle& anInitialParticle,
                                                     KSParticleQueue& aQueue ) = 0;

            //returns true if any of the state variables of aFinalParticle are changed
            virtual bool ExecutePostStepModification( KSParticle& anInitialParticle,
                                                     KSParticle& aFinalParticle,
                                                     KSParticleQueue& aQueue ) = 0;
    };

}

#endif
