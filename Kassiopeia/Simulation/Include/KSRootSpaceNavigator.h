#ifndef Kassiopeia_KSRootSpaceNavigator_h_
#define Kassiopeia_KSRootSpaceNavigator_h_

#include "KSSpaceNavigator.h"
#include "KSList.h"

#include "KSStep.h"
#include "KSTrajectory.h"

#include "KMathBracketingSolver.h"
using katrin::KMathBracketingSolver;

namespace Kassiopeia
{

    class KSRootSpaceNavigator :
        public KSComponentTemplate< KSRootSpaceNavigator, KSSpaceNavigator >
    {
        public:
            KSRootSpaceNavigator();
            KSRootSpaceNavigator( const KSRootSpaceNavigator& aCopy );
            KSRootSpaceNavigator* Clone() const;
            ~KSRootSpaceNavigator();

            //*****************
            //space interaction
            //*****************

        public:
            void CalculateNavigation( const KSTrajectory& aTrajectory, const KSParticle& aTrajectoryInitialParticle, const KSParticle& aTrajectoryFinalParticle, const KThreeVector& aTrajectoryCenter, const double& aTrajectoryRadius, const double& aTrajectoryStep, KSParticle& aNavigationParticle, double& aNavigationStep, bool& aNavigationFlag );
            void ExecuteNavigation( const KSParticle& anNavigationParticle, KSParticle& aFinalParticle, KSParticleQueue& aSecondaries ) const;
            void StartNavigation( KSParticle& aParticle, KSSpace* aRoot );
            void StopNavigation( KSParticle& aParticle, KSSpace* aRoot );

            //***********
            //composition
            //***********

        public:
            void SetSpaceNavigator( KSSpaceNavigator* anNavigation );
            void ClearSpaceNavigator( KSSpaceNavigator* anNavigation );

        private:
            KSSpaceNavigator* fSpaceNavigator;

            //******
            //action
            //******

        public:
            void SetStep( KSStep* anStep );
            void SetTrajectory( KSTrajectory* aTrajectory );

            void CalculateNavigation();
            void ExecuteNavigation();

        private:
            KSStep* fStep;
            const KSParticle* fTerminatorParticle;
            const KSParticle* fTrajectoryParticle;
            KSParticle* fNavigationParticle;
            KSParticle* fFinalParticle;
            KSParticleQueue* fParticleQueue;
            KSTrajectory* fTrajectory;
    };

}

#endif
