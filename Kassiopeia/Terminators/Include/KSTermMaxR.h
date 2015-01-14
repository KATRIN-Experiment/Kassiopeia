#ifndef Kassiopeia_KSTermMaxR_h_
#define Kassiopeia_KSTermMaxR_h_

#include "KSTerminator.h"

namespace Kassiopeia
{

    class KSParticle;

    class KSTermMaxR :
        public KSComponentTemplate< KSTermMaxR, KSTerminator >
    {
        public:
            KSTermMaxR();
            KSTermMaxR( const KSTermMaxR& aCopy );
            KSTermMaxR* Clone() const;
            virtual ~KSTermMaxR();

        public:
            void CalculateTermination( const KSParticle& anInitialParticle, bool& aFlag );
            void ExecuteTermination( const KSParticle& anInitialParticle, KSParticle& aFinalParticle, KSParticleQueue& aParticleQueue ) const;

        public:
            void SetMaxR( const double& aValue );

        private:
            double fMaxR;

    };

    inline void KSTermMaxR::SetMaxR( const double& aValue )
    {
        fMaxR = aValue;
    }

}

#endif
