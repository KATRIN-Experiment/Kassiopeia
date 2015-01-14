#ifndef Kassiopeia_KSTermMinLongEnergy_h_
#define Kassiopeia_KSTermMinLongEnergy_h_

#include "KSTerminator.h"

namespace Kassiopeia
{

    class KSParticle;

    class KSTermMinLongEnergy :
        public KSComponentTemplate< KSTermMinLongEnergy, KSTerminator >
    {
        public:
            KSTermMinLongEnergy();
            KSTermMinLongEnergy( const KSTermMinLongEnergy& aCopy );
            KSTermMinLongEnergy* Clone() const;
            virtual ~KSTermMinLongEnergy();

        public:
            void CalculateTermination( const KSParticle& anInitialParticle, bool& aFlag );
            void ExecuteTermination( const KSParticle& anInitialParticle, KSParticle& aFinalParticle, KSParticleQueue& aParticleQueue ) const;

        public:
            void SetMinLongEnergy( const double& aValue );

        private:
            double fMinLongEnergy;

    };

    inline void KSTermMinLongEnergy::SetMinLongEnergy( const double& aValue )
    {
        fMinLongEnergy = aValue;
    }

}

#endif
