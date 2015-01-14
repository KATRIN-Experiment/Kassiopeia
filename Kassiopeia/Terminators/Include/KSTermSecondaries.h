#ifndef Kassiopeia_KSTermSecondaries_h_
#define Kassiopeia_KSTermSecondaries_h_

#include "KSTerminator.h"

namespace Kassiopeia
{

    class KSTermSecondaries :
        public KSComponentTemplate<  KSTermSecondaries, KSTerminator >
    {
        public:
            KSTermSecondaries();
            KSTermSecondaries( const KSTermSecondaries& aCopy );
            KSTermSecondaries* Clone() const;
            virtual ~KSTermSecondaries();

        public:
            void CalculateTermination( const KSParticle& anInitialParticle, bool& aFlag );
            void ExecuteTermination( const KSParticle&, KSParticle& aFinalParticle, KSParticleQueue& ) const;

    };

}

#endif
