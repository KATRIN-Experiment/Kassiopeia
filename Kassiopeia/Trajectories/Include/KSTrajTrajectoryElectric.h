#ifndef Kassiopeia_KSTrajTrajectoryElectric_h_
#define Kassiopeia_KSTrajTrajectoryElectric_h_

#include "KSTrajectory.h"
#include "KSTrajElectricTypes.h"

#include "KSList.h"
#include "KField.h"

#include "KGBall.hh"
#include "KGBallSupportSet.hh"


namespace Kassiopeia
{

    class KSTrajTrajectoryElectric :
        public KSComponentTemplate< KSTrajTrajectoryElectric, KSTrajectory >,
        public KSTrajElectricDifferentiator
    {
        public:
            KSTrajTrajectoryElectric();
            KSTrajTrajectoryElectric( const KSTrajTrajectoryElectric& aCopy );
            KSTrajTrajectoryElectric* Clone() const;
            virtual ~KSTrajTrajectoryElectric();

        public:
            void SetIntegrator( KSTrajElectricIntegrator* anIntegrator );
            void ClearIntegrator( KSTrajElectricIntegrator* anIntegrator );

            void SetInterpolator( KSTrajElectricInterpolator* anInterpolator );
            void ClearInterpolator( KSTrajElectricInterpolator* anInterpolator );

            void AddTerm( KSTrajElectricDifferentiator* aTerm );
            void RemoveTerm( KSTrajElectricDifferentiator* aTerm );

            void AddControl( KSTrajElectricControl* aControl );
            void RemoveControl( KSTrajElectricControl* aControl );

            void SetReverseDirection( const bool& aFlag );
            const bool& GetReverseDirection() const;

            void SetAttemptLimit(unsigned int n)
            {
                if(n > 1 ){fMaxAttempts = n;}
                else{fMaxAttempts = 1;};
            }

            //**********
            //trajectory
            //**********

        public:
            void Reset();
            void CalculateTrajectory( const KSParticle& anInitialParticle, KSParticle& aFinalParticle, KThreeVector& aCenter, double& aRadius, double& aTimeStep );
            void ExecuteTrajectory( const double& aTimeStep, KSParticle& anIntermediateParticle ) const;
            void GetPiecewiseLinearApproximation(const KSParticle& anInitialParticle, const KSParticle& /*aFinalParticle*/, std::vector< KSParticle >* intermediateParticleStates) const;

            //********************
            //exact term interface
            //********************

            void SetPiecewiseTolerance(double ptol){fPiecewiseTolerance = ptol;};
            double GetPiecewiseTolerance() const {return fPiecewiseTolerance;};

            void SetMaxNumberOfSegments(double n_max){fNMaxSegments = n_max; if(fNMaxSegments < 1 ){fNMaxSegments = 1;};};
            unsigned int GetMaxNumberOfSegments() const {return fNMaxSegments;};

        public:
            virtual void Differentiate(double /*aTime*/, const KSTrajElectricParticle& aValue, KSTrajElectricDerivative& aDerivative ) const;

        private:
            KSTrajElectricParticle fInitialParticle;
            mutable KSTrajElectricParticle fIntermediateParticle;
            mutable KSTrajElectricParticle fFinalParticle;
            mutable KSTrajElectricError fError;

            KSTrajElectricIntegrator* fIntegrator;
            KSTrajElectricInterpolator* fInterpolator;
            KSList< KSTrajElectricDifferentiator > fTerms;
            KSList< KSTrajElectricControl > fControls;

            //piecewise linear approximation
            double fPiecewiseTolerance;
            unsigned int fNMaxSegments;
            mutable KGeoBag::KGBallSupportSet<3> fBallSupport;
            mutable std::vector<KSTrajElectricParticle> fIntermediateParticleStates;

            unsigned int fMaxAttempts;


    };

}

#endif
