#ifndef Kassiopeia_KSTrajTrajectoryAdiabatic_h_
#define Kassiopeia_KSTrajTrajectoryAdiabatic_h_

#include "KSTrajectory.h"
#include "KSTrajAdiabaticTypes.h"

#include "KSList.h"

#include "KGBall.hh"
#include "KGBallSupportSet.hh"

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

            void SetUseTruePosition( bool use){fUseTruePostion = use;};
            bool GetUseTruePosition() const {return fUseTruePostion;};

            void SetCyclotronFraction(double c_frac){fCyclotronFraction = c_frac;};
            double GetCyclotronFraction() const {return fCyclotronFraction;};

        public:
            void Reset();
            void CalculateTrajectory( const KSParticle& anInitialParticle, KSParticle& aFinalParticle, KThreeVector& aCenter, double& aRadius, double& aTimeStep );
            void ExecuteTrajectory( const double& aTimeStep, KSParticle& anIntermediateParticle ) const;
            void GetPiecewiseLinearApproximation(const KSParticle& anInitialParticle, const KSParticle& /*aFinalParticle*/, std::vector< KSParticle >* intermediateParticleStates) const;

            //********************
            //exact term interface
            //********************

        public:
            virtual void Differentiate(double aTime, const KSTrajAdiabaticParticle& aValue, KSTrajAdiabaticDerivative& aDerivative ) const;

        private:
            KSTrajAdiabaticParticle fInitialParticle;
            mutable KSTrajAdiabaticParticle fIntermediateParticle;
            mutable KSTrajAdiabaticParticle fFinalParticle;
            mutable KSTrajAdiabaticError fError;

            KSTrajAdiabaticIntegrator* fIntegrator;
            KSTrajAdiabaticInterpolator* fInterpolator;
            KSList< KSTrajAdiabaticDifferentiator > fTerms;
            KSList< KSTrajAdiabaticControl > fControls;

            //piecewise linear approximation
            double fPiecewiseTolerance;
            unsigned int fNMaxSegments;

            //if true, approximate the true particle's position for intermediate states
            bool fUseTruePostion;
            double fCyclotronFraction;

            mutable KGeoBag::KGBallSupportSet<3> fBallSupport;
            mutable std::vector<KSTrajAdiabaticParticle> fIntermediateParticleStates;
            unsigned int fMaxAttempts;
    };

}

#endif
