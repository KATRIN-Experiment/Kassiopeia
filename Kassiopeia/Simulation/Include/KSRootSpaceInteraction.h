#ifndef Kassiopeia_KSRootSpaceInteraction_h_
#define Kassiopeia_KSRootSpaceInteraction_h_

#include "KSSpaceInteraction.h"
#include "KSList.h"

#include "KSStep.h"
#include "KSTrajectory.h"

#include "KMathBracketingSolver.h"
using katrin::KMathBracketingSolver;

namespace Kassiopeia
{

    class KSRootSpaceInteraction :
        public KSComponentTemplate< KSRootSpaceInteraction, KSSpaceInteraction >
    {
        public:
            KSRootSpaceInteraction();
            KSRootSpaceInteraction( const KSRootSpaceInteraction& aCopy );
            KSRootSpaceInteraction* Clone() const;
            ~KSRootSpaceInteraction();

            //*****************
            //space interaction
            //*****************

        public:
            void CalculateInteraction( const KSTrajectory& aTrajectory, const KSParticle& aTrajectoryInitialParticle, const KSParticle& aTrajectoryFinalParticle, const KThreeVector& aTrajectoryCenter, const double& aTrajectoryRadius, const double& aTrajectoryTimeStep, KSParticle& anInteractionParticle, double& aTimeStep, bool& aFlag );
            void ExecuteInteraction( const KSParticle& anInteractionParticle, KSParticle& aFinalParticle, KSParticleQueue& aSecondaries ) const;

            //***********
            //composition
            //***********

        public:
            void AddSpaceInteraction( KSSpaceInteraction* anInteraction );
            void RemoveSpaceInteraction( KSSpaceInteraction* anInteraction );

        private:
            KSList< KSSpaceInteraction > fSpaceInteractions;
            KSSpaceInteraction* fSpaceInteraction;

            //******
            //action
            //******

        public:
            void SetStep( KSStep* anStep );
            void SetTrajectory( KSTrajectory* aTrajectory );

            void CalculateInteraction();
            void ExecuteInteraction();

            virtual void PushUpdateComponent();
            virtual void PushDeupdateComponent();

        private:
            KSStep* fStep;
            const KSParticle* fTerminatorParticle;
            const KSParticle* fTrajectoryParticle;
            KSParticle* fInteractionParticle;
            KSParticle* fFinalParticle;
            KSParticleQueue* fParticleQueue;
            KSTrajectory* fTrajectory;
    };

}

#endif
