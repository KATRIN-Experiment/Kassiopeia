#ifndef Kassiopeia_KSTermStepsize_h_
#define Kassiopeia_KSTermStepsize_h_

#include "KSTerminator.h"

#include "KField.h"

namespace Kassiopeia
{

    class KSParticle;

    class KSTermStepsize :
    public KSComponentTemplate< KSTermStepsize, KSTerminator >
    {
        public:
            KSTermStepsize();
            KSTermStepsize( const KSTermStepsize& aCopy );
            KSTermStepsize* Clone() const;
            virtual ~KSTermStepsize();

        public:
            void CalculateTermination( const KSParticle& anInitialParticle, bool& aFlag );
            void ExecuteTermination( const KSParticle& anInitialParticle, KSParticle& aFinalParticle, KSParticleQueue& aParticleQueue ) const;

        public:
            ;K_SET_GET( double, LowerLimit )
            ;K_SET_GET( double, UpperLimit )

        private:
            double fCurrentPathLength;

    };


}

#endif
