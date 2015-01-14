#ifndef Kassiopeia_KSTermMinEnergy_h_
#define Kassiopeia_KSTermMinEnergy_h_

#include "KSTerminator.h"

namespace Kassiopeia
{

    class KSParticle;

    class KSTermMinEnergy :
        public KSComponentTemplate< KSTermMinEnergy, KSTerminator >
    {
        public:
            KSTermMinEnergy();
            KSTermMinEnergy( const KSTermMinEnergy& aCopy );
            KSTermMinEnergy* Clone() const;
            virtual ~KSTermMinEnergy();

        public:
            void CalculateTermination( const KSParticle& anInitialParticle, bool& aFlag );
            void ExecuteTermination( const KSParticle& anInitialParticle, KSParticle& aFinalParticle, KSParticleQueue& aParticleQueue ) const;

        public:
            void SetMinEnergy( const double& aValue );

        private:
            double fMinEnergy;

    };

    inline void KSTermMinEnergy::SetMinEnergy( const double& aValue )
    {
        fMinEnergy = aValue;
    }

}

#endif
