#ifndef Kassiopeia_KSTrajTrajectoryMagnetic_h_
#define Kassiopeia_KSTrajTrajectoryMagnetic_h_

#include "KSTrajectory.h"
#include "KSTrajMagneticTypes.h"

#include "KSList.h"
#include "KField.h"

namespace Kassiopeia
{

    class KSTrajTrajectoryMagnetic :
        public KSComponentTemplate< KSTrajTrajectoryMagnetic, KSTrajectory >,
        public KSTrajMagneticDifferentiator
    {
        public:
            KSTrajTrajectoryMagnetic();
            KSTrajTrajectoryMagnetic( const KSTrajTrajectoryMagnetic& aCopy );
            KSTrajTrajectoryMagnetic* Clone() const;
            virtual ~KSTrajTrajectoryMagnetic();

        public:
            void SetIntegrator( KSTrajMagneticIntegrator* anIntegrator );
            void ClearIntegrator( KSTrajMagneticIntegrator* anIntegrator );

            void SetInterpolator( KSTrajMagneticInterpolator* anInterpolator );
            void ClearInterpolator( KSTrajMagneticInterpolator* anInterpolator );

            void AddTerm( KSTrajMagneticDifferentiator* aTerm );
            void RemoveTerm( KSTrajMagneticDifferentiator* aTerm );

            void AddControl( KSTrajMagneticControl* aControl );
            void RemoveControl( KSTrajMagneticControl* aControl );

            void SetReverseDirection( const bool& aFlag );
            const bool& GetReverseDirection() const;

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
            virtual void Differentiate( const KSTrajMagneticParticle& aValue, KSTrajMagneticDerivative& aDerivative ) const;

        private:
            KSTrajMagneticParticle fInitialParticle;
            mutable KSTrajMagneticParticle fIntermediateParticle;
            mutable KSTrajMagneticParticle fFinalParticle;
            mutable KSTrajMagneticError fError;

            KSTrajMagneticIntegrator* fIntegrator;
            KSTrajMagneticInterpolator* fInterpolator;
            KSList< KSTrajMagneticDifferentiator > fTerms;
            KSList< KSTrajMagneticControl > fControls;

            bool fReverseDirection;
    };

}

#endif

