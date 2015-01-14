#ifndef Kassiopeia_KSTermMaxSteps_h_
#define Kassiopeia_KSTermMaxSteps_h_

#include "KSTerminator.h"

namespace Kassiopeia
{

    class KSTermMaxSteps :
        public KSComponentTemplate< KSTermMaxSteps, KSTerminator >
    {
        public:
            KSTermMaxSteps();
            KSTermMaxSteps( const KSTermMaxSteps& aCopy );
            KSTermMaxSteps* Clone() const;
            virtual ~KSTermMaxSteps();

        public:
            void CalculateTermination( const KSParticle& anInitialParticle, bool& aFlag );
            void ExecuteTermination( const KSParticle& anInitialParticle, KSParticle& aFinalParticle, KSParticleQueue& aParticleQueue ) const;

        public:
            void SetMaxSteps( const unsigned int& maxsteps );

        protected:
            virtual void ActivateComponent();
            virtual void DeactivateComponent();

        private:
            unsigned int fMaxSteps;
            unsigned int fSteps;
    };

    inline void KSTermMaxSteps::SetMaxSteps( const unsigned int& maxsteps )
    {
        fMaxSteps = maxsteps;
        return;
    }

}

#endif
