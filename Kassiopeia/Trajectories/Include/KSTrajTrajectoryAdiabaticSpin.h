#ifndef Kassiopeia_KSTrajTrajectoryAdiabaticSpin_h_
#define Kassiopeia_KSTrajTrajectoryAdiabaticSpin_h_

#include "KSTrajectory.h"
#include "KSTrajAdiabaticSpinTypes.h"

#include "KSList.h"

#include "KGBall.hh"
#include "KGBallSupportSet.hh"

namespace Kassiopeia
{

    class KSTrajTrajectoryAdiabaticSpin :
        public KSComponentTemplate< KSTrajTrajectoryAdiabaticSpin, KSTrajectory >,
        public KSTrajAdiabaticSpinDifferentiator
    {
        public:
            KSTrajTrajectoryAdiabaticSpin();
            KSTrajTrajectoryAdiabaticSpin( const KSTrajTrajectoryAdiabaticSpin& aCopy );
            KSTrajTrajectoryAdiabaticSpin* Clone() const;
            virtual ~KSTrajTrajectoryAdiabaticSpin();

        public:
            void SetIntegrator( KSTrajAdiabaticSpinIntegrator* anIntegrator );
            void ClearIntegrator( KSTrajAdiabaticSpinIntegrator* anIntegrator );

            void SetInterpolator( KSTrajAdiabaticSpinInterpolator* anInterpolator );
            void ClearInterpolator( KSTrajAdiabaticSpinInterpolator* anInterpolator );

            void AddTerm( KSTrajAdiabaticSpinDifferentiator* aTerm );
            void RemoveTerm( KSTrajAdiabaticSpinDifferentiator* aTerm );

            void AddControl( KSTrajAdiabaticSpinControl* aControl );
            void RemoveControl( KSTrajAdiabaticSpinControl* aControlSize );

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
            //adiabatic spin term interface
            //********************

        public:
            virtual void Differentiate(double aTime, const KSTrajAdiabaticSpinParticle& aValue, KSTrajAdiabaticSpinDerivative& aDerivative ) const;

        private:

            KSTrajAdiabaticSpinParticle fInitialParticle;
            mutable KSTrajAdiabaticSpinParticle fIntermediateParticle;
            mutable KSTrajAdiabaticSpinParticle fFinalParticle;
            mutable KSTrajAdiabaticSpinError fError;

            KSTrajAdiabaticSpinIntegrator* fIntegrator;
            KSTrajAdiabaticSpinInterpolator* fInterpolator;
            KSList< KSTrajAdiabaticSpinDifferentiator > fTerms;
            KSList< KSTrajAdiabaticSpinControl > fControls;

            //piecewise linear approximation
            double fPiecewiseTolerance;
            unsigned int fNMaxSegments;
            mutable KGeoBag::KGBallSupportSet<3> fBallSupport;
            mutable std::vector<KSTrajAdiabaticSpinParticle> fIntermediateParticleStates;

            unsigned int fMaxAttempts;

    };

}

#endif
