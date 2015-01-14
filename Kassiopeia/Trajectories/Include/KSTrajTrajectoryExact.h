#ifndef Kassiopeia_KSTrajTrajectoryExact_h_
#define Kassiopeia_KSTrajTrajectoryExact_h_

#include "KSTrajectory.h"
#include "KSTrajExactTypes.h"

#include "KSList.h"

namespace Kassiopeia
{

    class KSTrajTrajectoryExact :
        public KSComponentTemplate< KSTrajTrajectoryExact, KSTrajectory >,
        public KSTrajExactDifferentiator
    {
        public:
            KSTrajTrajectoryExact();
            KSTrajTrajectoryExact( const KSTrajTrajectoryExact& aCopy );
            KSTrajTrajectoryExact* Clone() const;
            virtual ~KSTrajTrajectoryExact();

        public:
            void SetIntegrator( KSTrajExactIntegrator* anIntegrator );
            void ClearIntegrator( KSTrajExactIntegrator* anIntegrator );

            void SetInterpolator( KSTrajExactInterpolator* anInterpolator );
            void ClearInterpolator( KSTrajExactInterpolator* anInterpolator );

            void AddTerm( KSTrajExactDifferentiator* aTerm );
            void RemoveTerm( KSTrajExactDifferentiator* aTerm );

            void AddControl( KSTrajExactControl* aControl );
            void RemoveControl( KSTrajExactControl* aControlSize );

            //**********
            //trajectory
            //**********

        public:
            void CalculateTrajectory( const KSParticle& anInitialParticle, KSParticle& aFinalParticle, KThreeVector& aCenter, double& aRadius, double& aTimeStep );
            void ExecuteTrajectory( const double& aTimeStep, KSParticle& anIntermediateParticle ) const;

            //********************
            //exact term interface
            //********************

        public:
            virtual void Differentiate( const KSTrajExactParticle& aValue, KSTrajExactDerivative& aDerivative ) const;

        private:
            KSTrajExactParticle fInitialParticle;
            mutable KSTrajExactParticle fIntermediateParticle;
            mutable KSTrajExactParticle fFinalParticle;
            mutable KSTrajExactError fError;

            KSTrajExactIntegrator* fIntegrator;
            KSTrajExactInterpolator* fInterpolator;
            KSList< KSTrajExactDifferentiator > fTerms;
            KSList< KSTrajExactControl > fControls;
    };

}

#endif
