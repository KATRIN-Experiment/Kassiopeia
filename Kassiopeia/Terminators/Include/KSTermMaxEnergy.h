#ifndef Kassiopeia_KSTermMaxEnergy_h_
#define Kassiopeia_KSTermMaxEnergy_h_

#include "KSTerminator.h"

namespace Kassiopeia
{

    class KSParticle;

    class KSTermMaxEnergy :
        public KSComponentTemplate< KSTermMaxEnergy, KSTerminator >
    {
        public:
            KSTermMaxEnergy();
            KSTermMaxEnergy( const KSTermMaxEnergy& aCopy );
            KSTermMaxEnergy* Clone() const;
            virtual ~KSTermMaxEnergy();

        public:
            void CalculateTermination( const KSParticle& anInitialParticle, bool& aFlag );
            void ExecuteTermination( const KSParticle& anInitialParticle, KSParticle& aFinalParticle, KSParticleQueue& aParticleQueue ) const;

        public:
            void SetMaxEnergy( const double& aValue );

        private:
            double fMaxEnergy;

    };

    inline void KSTermMaxEnergy::SetMaxEnergy( const double& aValue )
    {
        fMaxEnergy = aValue;
    }

}

#endif
