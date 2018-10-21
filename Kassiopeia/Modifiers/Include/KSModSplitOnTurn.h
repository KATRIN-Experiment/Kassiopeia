#ifndef Kassiopeia_KSModSplitOnTurn_h_
#define Kassiopeia_KSModSplitOnTurn_h_

#include "KField.h"
#include "KSStepModifier.h"
#include "KSComponentTemplate.h"

namespace Kassiopeia
{

    class KSModSplitOnTurn :
            public KSComponentTemplate< KSModSplitOnTurn, KSStepModifier >
    {
        public:
            enum {
                // use binary numbers here (allows combinations like `eForward | eBackward`)
                eForward    = 0b0001,
                eBackward   = 0b0010,
            };

        public:
            KSModSplitOnTurn();
            KSModSplitOnTurn( const KSModSplitOnTurn& aCopy);
            KSModSplitOnTurn* Clone() const;
            virtual ~KSModSplitOnTurn();

        public:
            bool ExecutePreStepModification( KSParticle& anInitialParticle, KSParticleQueue& aQueue );
            bool ExecutePostStepModification( KSParticle& anInitialParticle, KSParticle& aFinalParticle, KSParticleQueue& aQueue );

        public:
            K_SET_GET( int, Direction );

        private:
            double fCurrentDotProduct;

        private:
            void InitializeComponent();
            void DeinitializeComponent();

        protected:
            virtual void PullDeupdateComponent();
            virtual void PushDeupdateComponent();
    };
}

#endif
