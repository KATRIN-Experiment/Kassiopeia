#ifndef Kassiopeia_KSIntDensity_h_
#define Kassiopeia_KSIntDensity_h_

#include "KSComponentTemplate.h"

#include "KSParticle.h"

namespace Kassiopeia
{
    class KSStep;

    class KSIntDensity:
        public KSComponentTemplate< KSIntDensity >
    {
        public:
    		KSIntDensity();
            virtual ~KSIntDensity();
            virtual KSIntDensity* Clone() const = 0;

        public:
            virtual void CalculateDensity( const KSParticle& aParticle, double& aDensity ) = 0;
    };

}

#endif

