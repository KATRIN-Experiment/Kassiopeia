#ifndef Kassiopeia_KSTermMaxStepTime_h_
#define Kassiopeia_KSTermMaxStepTime_h_

#include "KSTerminator.h"

#include <ctime>

namespace Kassiopeia
{

    class KSParticle;

    class KSTermMaxStepTime :
        public KSComponentTemplate< KSTermMaxStepTime, KSTerminator >
    {
        public:
            KSTermMaxStepTime();
            KSTermMaxStepTime( const KSTermMaxStepTime& aCopy );
            KSTermMaxStepTime* Clone() const;
            virtual ~KSTermMaxStepTime();

        public:
            void CalculateTermination( const KSParticle& anInitialParticle, bool& aFlag );
            void ExecuteTermination( const KSParticle& anInitialParticle, KSParticle& aFinalParticle, KSParticleQueue& aParticleQueue ) const;

        public:
            void SetTime( const double& aValue );

        private:
            double fTime;
            std::clock_t fLastClock;
    };

    inline void KSTermMaxStepTime::SetTime( const double& aValue )
    {
        fTime = aValue;
    }

}

#endif
