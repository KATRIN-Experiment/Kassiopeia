#ifndef Kassiopeia_KSRootSurfaceNavigator_h_
#define Kassiopeia_KSRootSurfaceNavigator_h_

#include "KSSurfaceNavigator.h"
#include "KSList.h"

#include "KSStep.h"
#include "KSTrajectory.h"

#include "KMathBracketingSolver.h"
using katrin::KMathBracketingSolver;

namespace Kassiopeia
{

    class KSRootSurfaceNavigator :
        public KSComponentTemplate< KSRootSurfaceNavigator, KSSurfaceNavigator >
    {
        public:
            KSRootSurfaceNavigator();
            KSRootSurfaceNavigator( const KSRootSurfaceNavigator& aCopy );
            KSRootSurfaceNavigator* Clone() const;
            ~KSRootSurfaceNavigator();

            //******************
            //surface navigation
            //******************

        public:
            void ExecuteNavigation( const KSParticle& anInitialParticle, const KSParticle& anNavigationParticle, KSParticle& aFinalParticle, KSParticleQueue& aSecondaries ) const;
            void FinalizeNavigation( KSParticle& aFinalParticle ) const;

            //***********
            //composition
            //***********

        public:
            void SetSurfaceNavigator( KSSurfaceNavigator* anNavigation );
            void ClearSurfaceNavigator( KSSurfaceNavigator* anNavigation );

        private:
            KSSurfaceNavigator* fSurfaceNavigator;

            //******
            //action
            //******

        public:
            void SetStep( KSStep* anStep );

            void ExecuteNavigation();
            void FinalizeNavigation();

        private:
            KSStep* fStep;
            const KSParticle* fTerminatorParticle;
            const KSParticle* fInteractionParticle;
            KSParticle* fFinalParticle;
            KSParticleQueue* fParticleQueue;
    };

}

#endif
