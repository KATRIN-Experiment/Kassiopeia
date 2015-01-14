#ifndef Kassiopeia_KSTrajTrajectoryLinear_h_
#define Kassiopeia_KSTrajTrajectoryLinear_h_

#include "KSTrajectory.h"

namespace Kassiopeia
{

    class KSTrajTrajectoryLinear :
        public KSComponentTemplate< KSTrajTrajectoryLinear, KSTrajectory >
    {
        public:
        	KSTrajTrajectoryLinear();
        	KSTrajTrajectoryLinear( const KSTrajTrajectoryLinear& aCopy );
        	KSTrajTrajectoryLinear* Clone() const;
            virtual ~KSTrajTrajectoryLinear();

        public:
            void SetLength( const double& aLength );
            const double& GetLength() const;

        private:
            double fLength;

            //**********
            //trajectory
            //**********

        public:
            void CalculateTrajectory( const KSParticle& anInitialParticle, KSParticle& aFinalParticle, KThreeVector& aCenter, double& aRadius, double& aTimeStep );
            void ExecuteTrajectory( const double& aTimeStep, KSParticle& anIntermediateParticle ) const;

        private:
            double fTime;
            KThreeVector fPosition;
            KThreeVector fVelocity;
    };

}

#endif
