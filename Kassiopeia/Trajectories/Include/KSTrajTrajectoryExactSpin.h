#ifndef Kassiopeia_KSTrajTrajectoryExactSpin_h_
#define Kassiopeia_KSTrajTrajectoryExactSpin_h_

#include "KSTrajectory.h"
#include "KSTrajExactSpinTypes.h"

#include "KSList.h"

#include "KGBall.hh"
#include "KGBallSupportSet.hh"

namespace Kassiopeia
{

    class KSTrajTrajectoryExactSpin :
        public KSComponentTemplate< KSTrajTrajectoryExactSpin, KSTrajectory >,
        public KSTrajExactSpinDifferentiator
    {
        public:
            KSTrajTrajectoryExactSpin();
            KSTrajTrajectoryExactSpin( const KSTrajTrajectoryExactSpin& aCopy );
            KSTrajTrajectoryExactSpin* Clone() const;
            virtual ~KSTrajTrajectoryExactSpin();

        public:
            void SetIntegrator( KSTrajExactSpinIntegrator* anIntegrator );
            void ClearIntegrator( KSTrajExactSpinIntegrator* anIntegrator );

            void SetInterpolator( KSTrajExactSpinInterpolator* anInterpolator );
            void ClearInterpolator( KSTrajExactSpinInterpolator* anInterpolator );

            void AddTerm( KSTrajExactSpinDifferentiator* aTerm );
            void RemoveTerm( KSTrajExactSpinDifferentiator* aTerm );

            void AddControl( KSTrajExactSpinControl* aControl );
            void RemoveControl( KSTrajExactSpinControl* aControlSize );

            void SetAttemptLimit(unsigned int n)
            {
                if(n > 1 ){fMaxAttempts = n;}
                else{fMaxAttempts = 1;};
            }

            //**********
            //trajectory
            //**********

            void SetPiecewiseTolerance(double ptol){fPiecewiseTolerance = ptol;};
            double GetPiecewiseTolerance() const {return fPiecewiseTolerance;};

            void SetMaxNumberOfSegments(double n_max){fNMaxSegments = n_max; if(fNMaxSegments < 1 ){fNMaxSegments = 1;};};
            unsigned int GetMaxNumberOfSegments() const {return fNMaxSegments;};

        public:

            void Reset();
            void CalculateTrajectory( const KSParticle& anInitialParticle, KSParticle& aFinalParticle, KThreeVector& aCenter, double& aRadius, double& aTimeStep );
            void ExecuteTrajectory( const double& aTimeStep, KSParticle& anIntermediateParticle ) const;
            void GetPiecewiseLinearApproximation(const KSParticle& anInitialParticle, const KSParticle& /*aFinalParticle*/, std::vector< KSParticle >* intermediateParticleStates) const;

            //********************
            //ExactSpin term interface
            //********************

        public:
            virtual void Differentiate(double aTime, const KSTrajExactSpinParticle& aValue, KSTrajExactSpinDerivative& aDerivative ) const;

        private:

            KSTrajExactSpinParticle fInitialParticle;
            mutable KSTrajExactSpinParticle fIntermediateParticle;
            mutable KSTrajExactSpinParticle fFinalParticle;
            mutable KSTrajExactSpinError fError;

            KSTrajExactSpinIntegrator* fIntegrator;
            KSTrajExactSpinInterpolator* fInterpolator;
            KSList< KSTrajExactSpinDifferentiator > fTerms;
            KSList< KSTrajExactSpinControl > fControls;

            //piecewise linear approximation
            double fPiecewiseTolerance;
            unsigned int fNMaxSegments;
            mutable KGeoBag::KGBallSupportSet<3> fBallSupport;
            mutable std::vector<KSTrajExactSpinParticle> fIntermediateParticleStates;

            unsigned int fMaxAttempts;

    };

}

#endif
