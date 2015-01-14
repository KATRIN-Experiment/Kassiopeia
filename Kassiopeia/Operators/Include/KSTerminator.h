#ifndef Kassiopeia_KSTerminator_h_
#define Kassiopeia_KSTerminator_h_

#include "KSComponentTemplate.h"
#include "KSParticle.h"

namespace Kassiopeia
{

    class KSTrack;

    class KSTerminator:
        public KSComponentTemplate<  KSTerminator >
    {
        public:
            KSTerminator();
            virtual ~KSTerminator();

        public:
            virtual void CalculateTermination( const KSParticle& anInitialParticle, bool& aFlag ) = 0;
            virtual void ExecuteTermination( const KSParticle& anInitialParticle, KSParticle& aFinalParticle, KSParticleQueue& aQueue ) const = 0;
    };


}

#endif
