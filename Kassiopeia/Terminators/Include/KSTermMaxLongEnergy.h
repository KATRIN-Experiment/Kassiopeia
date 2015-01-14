#ifndef Kassiopeia_KSTermMaxLongEnergy_h_
#define Kassiopeia_KSTermMaxLongEnergy_h_

#include "KSTerminator.h"

namespace Kassiopeia
{

    class KSParticle;

    class KSTermMaxLongEnergy :
        public KSComponentTemplate< KSTermMaxLongEnergy, KSTerminator >
    {
        public:
            KSTermMaxLongEnergy();
            KSTermMaxLongEnergy( const KSTermMaxLongEnergy& aCopy );
            KSTermMaxLongEnergy* Clone() const;
            virtual ~KSTermMaxLongEnergy();

        public:
            void CalculateTermination( const KSParticle& anInitialParticle, bool& aFlag );
            void ExecuteTermination( const KSParticle& anInitialParticle, KSParticle& aFinalParticle, KSParticleQueue& aParticleQueue ) const;

        public:
            void SetMaxLongEnergy( const double& aValue );

        private:
            double fMaxLongEnergy;

    };

    inline void KSTermMaxLongEnergy::SetMaxLongEnergy( const double& aValue )
    {
        fMaxLongEnergy = aValue;
    }

}

#endif
