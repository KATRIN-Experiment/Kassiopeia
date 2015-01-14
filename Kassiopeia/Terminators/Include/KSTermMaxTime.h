#ifndef Kassiopeia_KSTermMaxTime_h_
#define Kassiopeia_KSTermMaxTime_h_

#include "KSTerminator.h"

namespace Kassiopeia
{

    class KSParticle;

    class KSTermMaxTime :
        public KSComponentTemplate< KSTermMaxTime, KSTerminator >
    {
        public:
            KSTermMaxTime();
            KSTermMaxTime( const KSTermMaxTime& aCopy );
            KSTermMaxTime* Clone() const;
            virtual ~KSTermMaxTime();

        public:
            void CalculateTermination( const KSParticle& anInitialParticle, bool& aFlag );
            void ExecuteTermination( const KSParticle& anInitialParticle, KSParticle& aFinalParticle, KSParticleQueue& aParticleQueue ) const;

        public:
            void SetTime( const double& aValue );

        private:
            double fTime;
    };

    inline void KSTermMaxTime::SetTime( const double& aValue )
    {
        fTime = aValue;
    }

}

#endif
