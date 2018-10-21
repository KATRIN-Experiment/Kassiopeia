#ifndef Kassiopeia_KSTermMagnetron_h_
#define Kassiopeia_KSTermMagnetron_h_

#include "KSTerminator.h"

namespace Kassiopeia
{

    class KSParticle;

    class KSTermMagnetron :
        public KSComponentTemplate< KSTermMagnetron, KSTerminator >
    {
        public:
            KSTermMagnetron();
            KSTermMagnetron( const KSTermMagnetron& aCopy );
            KSTermMagnetron* Clone() const;
            virtual ~KSTermMagnetron();

        public:
            void CalculateTermination( const KSParticle& anInitialParticle, bool& aFlag );
            void ExecuteTermination( const KSParticle& anInitialParticle, KSParticle& aFinalParticle, KSParticleQueue& aParticleQueue ) const;

        public:
            void SetMaxPhi( const double& aValue );

        private:
            double fMaxPhi;
            bool fFirstStep;
            double fPhiFirstStep;
            KThreeVector fPositionBefore;
            bool fAtanJump;
            unsigned int fJumpDirection; //1 = 0->2pi, 2 = 2pi->0

    };

    inline void KSTermMagnetron::SetMaxPhi( const double& aValue )
    {
        fMaxPhi = aValue;
    }

}

#endif
