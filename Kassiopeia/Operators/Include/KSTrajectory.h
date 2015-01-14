#ifndef Kassiopeia_KSTrajectory_h_
#define Kassiopeia_KSTrajectory_h_

#include "KSComponentTemplate.h"
#include "KSParticle.h"

namespace Kassiopeia
{

    class KSTrajectory :
    	public KSComponentTemplate< KSTrajectory >
    {
        public:
            KSTrajectory();
            virtual ~KSTrajectory();

        public:
            virtual void CalculateTrajectory(
                const KSParticle& anInitialParticle,
                KSParticle& aFinalParticle,
                KThreeVector& aCenter,
                double& aRadius,
                double& aTimeStep
            ) = 0;

            virtual void ExecuteTrajectory(
                const double& aTimeStep,
                KSParticle& anIntermediateParticle
            ) const = 0;
    };

}

#endif
