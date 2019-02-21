#ifndef Kassiopeia_KSIntSpinFlipPulse_h_
#define Kassiopeia_KSIntSpinFlipPulse_h_

#include "KSSpaceInteraction.h"

#include <vector>
using std::vector;

namespace Kassiopeia
{

    class KSIntSpinFlipPulse :
        public KSComponentTemplate< KSIntSpinFlipPulse, KSSpaceInteraction >
    {
        public:
            KSIntSpinFlipPulse();
            KSIntSpinFlipPulse( const KSIntSpinFlipPulse& aCopy );
            KSIntSpinFlipPulse* Clone() const;
            virtual ~KSIntSpinFlipPulse();

        public:
            void CalculateInteraction(
                    const KSTrajectory& aTrajectory,
                    const KSParticle& aTrajectoryInitialParticle,
                    const KSParticle& aTrajectoryFinalParticle,
                    const KThreeVector& aTrajectoryCenter,
                    const double& aTrajectoryRadius,
                    const double& aTrajectoryTimeStep,
                    KSParticle& anInteractionParticle,
                    double& aTimeStep,
                    bool& aFlag
            );

            void ExecuteInteraction(
                    const KSParticle& anInteractionParticle,
                    KSParticle& aFinalParticle,
                    KSParticleQueue& aSecondaries
            ) const;

            //***********
            //composition
            //***********

        public:
            void SetTime( const double& aTime );

        private:
            mutable bool fDone;
            double fTime;

            //**************
            //initialization
            //**************

        // protected:
        //     virtual void PushUpdateComponent();
        //     virtual void PushDeupdateComponent();
    };

}

#endif
