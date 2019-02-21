#ifndef Kassiopeia_KSTermMaxTotalTime_h_
#define Kassiopeia_KSTermMaxTotalTime_h_

#include "KSTerminator.h"

#include <ctime>

namespace Kassiopeia
{

    class KSParticle;

    class KSTermMaxTotalTime :
        public KSComponentTemplate< KSTermMaxTotalTime, KSTerminator >
    {
        public:
            KSTermMaxTotalTime();
            KSTermMaxTotalTime( const KSTermMaxTotalTime& aCopy );
            KSTermMaxTotalTime* Clone() const;
            virtual ~KSTermMaxTotalTime();

        public:
            void CalculateTermination( const KSParticle& anInitialParticle, bool& aFlag );
            void ExecuteTermination( const KSParticle& anInitialParticle, KSParticle& aFinalParticle, KSParticleQueue& aParticleQueue ) const;

        public:
            void SetTime( const double& aValue );

        private:
            double fTime;
    };

    inline void KSTermMaxTotalTime::SetTime( const double& aValue )
    {
        fTime = aValue;
    }

}

#endif
