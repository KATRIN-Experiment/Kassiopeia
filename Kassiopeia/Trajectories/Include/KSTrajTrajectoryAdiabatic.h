#ifndef Kassiopeia_KSTrajTrajectoryAdiabatic_h_
#define Kassiopeia_KSTrajTrajectoryAdiabatic_h_

#include "KSTrajectory.h"
#include "KSTrajAdiabaticTypes.h"

#include "KSList.h"

namespace Kassiopeia
{

    class KSTrajTrajectoryAdiabatic :
        public KSComponentTemplate< KSTrajTrajectoryAdiabatic, KSTrajectory >,
        public KSTrajAdiabaticDifferentiator
    {
        public:
            KSTrajTrajectoryAdiabatic();
            KSTrajTrajectoryAdiabatic( const KSTrajTrajectoryAdiabatic& aCopy );
            KSTrajTrajectoryAdiabatic* Clone() const;
            virtual ~KSTrajTrajectoryAdiabatic();

        public:
            void SetIntegrator( KSTrajAdiabaticIntegrator* anIntegrator );
            void ClearIntegrator( KSTrajAdiabaticIntegrator* anIntegrator );

            void SetInterpolator( KSTrajAdiabaticInterpolator* anInterpolator );
            void ClearInterpolator( KSTrajAdiabaticInterpolator* anInterpolator );

            void AddTerm( KSTrajAdiabaticDifferentiator* aTerm );
            void RemoveTerm( KSTrajAdiabaticDifferentiator* aTerm );

            void AddControl( KSTrajAdiabaticControl* aControl );
            void RemoveControl( KSTrajAdiabaticControl* aControl );

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
            virtual void Differentiate( const KSTrajAdiabaticParticle& aValue, KSTrajAdiabaticDerivative& aDerivative ) const;

        private:
            KSTrajAdiabaticParticle fInitialParticle;
            mutable KSTrajAdiabaticParticle fIntermediateParticle;
            mutable KSTrajAdiabaticParticle fFinalParticle;
            mutable KSTrajAdiabaticError fError;

            KSTrajAdiabaticIntegrator* fIntegrator;
            KSTrajAdiabaticInterpolator* fInterpolator;
            KSList< KSTrajAdiabaticDifferentiator > fTerms;
            KSList< KSTrajAdiabaticControl > fControls;
    };

}

#endif
