#ifndef _Kassiopeia_KSRootSurfaceInteraction_h_
#define _Kassiopeia_KSRootSurfaceInteraction_h_

#include "KSSurfaceInteraction.h"
#include "KSStep.h"
#include "KSList.h"

namespace Kassiopeia
{

    class KSRootSurfaceInteraction :
        public KSComponentTemplate< KSRootSurfaceInteraction, KSSurfaceInteraction >
    {
        public:
            KSRootSurfaceInteraction();
            KSRootSurfaceInteraction( const KSRootSurfaceInteraction& aCopy );
            KSRootSurfaceInteraction* Clone() const;
            ~KSRootSurfaceInteraction();

            //*******************
            //surface interaction
            //*******************

        public:
            void ExecuteInteraction( const KSParticle& anInitialParticle, KSParticle& aFinalParticle, KSParticleQueue& aSecondaries );

            //***********
            //composition
            //***********

        public:
            void SetSurfaceInteraction( KSSurfaceInteraction* anInteraction );
            void ClearSurfaceInteraction( KSSurfaceInteraction* anInteraction );

        private:
            KSSurfaceInteraction* fSurfaceInteraction;

            //******
            //action
            //******

        public:
            void SetStep( KSStep* anStep );

            void ExecuteInteraction();

        private:
            KSStep* fStep;
            const KSParticle* fTerminatorParticle;
            KSParticle* fInteractionParticle;
            KSParticleQueue* fParticleQueue;
    };

}

#endif

