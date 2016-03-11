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

            void Reset();
            void CalculateTrajectory( const KSParticle& anInitialParticle, KSParticle& aFinalParticle, KThreeVector& aCenter, double& aRadius, double& aTimeStep );
            void ExecuteTrajectory( const double& aTimeStep, KSParticle& anIntermediateParticle ) const;
            void GetPiecewiseLinearApproximation(const KSParticle& anInitialParticle, const KSParticle& aFinalParticle, std::vector< KSParticle >* intermediateParticleStates) const;


        private:
            double fTime;
            KThreeVector fPosition;
            KThreeVector fVelocity;

            //internal state for piecewise approximation
            KSParticle fFirstParticle;
            KSParticle fLastParticle;
    };

}

#endif
