#ifndef Kassiopeia_KSTermTrapped_h_
#define Kassiopeia_KSTermTrapped_h_

#include "KSTerminator.h"
#include "KField.h"

namespace Kassiopeia
{

    class KSParticle;

    class KSTermTrapped :
        public KSComponentTemplate< KSTermTrapped, KSTerminator >
    {
        public:
    		KSTermTrapped();
    		KSTermTrapped( const KSTermTrapped& aCopy );
    		KSTermTrapped* Clone() const;
            virtual ~KSTermTrapped();

            K_SET_GET(int, MaxTurns)

        public:
            void CalculateTermination( const KSParticle& anInitialParticle, bool& aFlag );
            void ExecuteTermination( const KSParticle& anInitialParticle, KSParticle& aFinalParticle, KSParticleQueue& aParticleQueue ) const;

        protected:
            virtual void ActivateComponent();
            virtual void DeactivateComponent();

        private:
            int fCurrentTurns;
            double fCurrentDotProduct;
    };

}

#endif
