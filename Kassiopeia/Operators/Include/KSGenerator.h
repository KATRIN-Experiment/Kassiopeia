#ifndef Kassiopeia_KSGenerator_h_
#define Kassiopeia_KSGenerator_h_

#include "KSComponentTemplate.h"
#include "KSParticle.h"

namespace Kassiopeia
{

    class KSGenerator :
        public KSComponentTemplate< KSGenerator >
    {
        public:
            KSGenerator();
            virtual ~KSGenerator();

        public:
            virtual void ExecuteGeneration( KSParticleQueue& aPrimaries ) = 0;
    };

}

#endif
