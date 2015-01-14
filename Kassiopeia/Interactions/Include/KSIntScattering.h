#ifndef Kassiopeia_KSIntScattering_h_
#define Kassiopeia_KSIntScattering_h_

#include "KSSpaceInteraction.h"
#include "KSIntDensity.h"
#include "KSIntCalculator.h"

#include <vector>
using std::vector;

namespace Kassiopeia
{

    class KSIntCalculator;

    class KSIntScattering :
        public KSComponentTemplate< KSIntScattering, KSSpaceInteraction >
    {
        public:
            KSIntScattering();
            KSIntScattering( const KSIntScattering& aCopy );
            KSIntScattering* Clone() const;
            virtual ~KSIntScattering();

        public:
            void CalculateAverageCrossSection(
                    const KSParticle& aTrajectoryInitialParticle,
                    const KSParticle& aTrajectoryFinalParticle,
                    double& anAverageCrossSection
            );

            void DiceCalculator( const double& anAverageCrossSection );

            void CalculateInteraction(
                    const KSTrajectory& aTrajectory,
                    const KSParticle& aTrajectoryInitialParticle,
                    const KSParticle& aTrajectoryFinalParticle,
                    const KThreeVector& aTrajectoryCenter,
                    const double& aTrajectoryRadius,
                    const double& aTrajectoryTimeStep,
                    KSParticle& anInteractionParticle,
                    double& aTimeStep,
                    bool& aFlag
            );

            void ExecuteInteraction(
                    const KSParticle& anInteractionParticle,
                    KSParticle& aFinalParticle,
                    KSParticleQueue& aSecondaries
            ) const;

            //***********
            //composition
            //***********

        public:
            void SetSplit( const bool& aSplit );
            const bool& GetSplit() const;

            void SetDensity( KSIntDensity* const aDensityCalculator );
            void ClearDensity( KSIntDensity* const aDensityCalculator );

            void AddCalculator( KSIntCalculator* const aScatteringCalculator );
            void RemoveCalculator( KSIntCalculator* const aScatteringCalculator );

            void SetEnhancement( double anEnhancement );

        private:
            bool fSplit;
            KSIntDensity* fDensity;
            KSIntCalculator* fCalculator;
            vector< KSIntCalculator* > fCalculators;
            vector< double > fCrossSections;

            double fEnhancement;

            //**************
            //initialization
            //**************

        protected:
            virtual void InitializeComponent();
            virtual void ActivateComponent();
            virtual void DeinitializeComponent();
            virtual void DeactivateComponent();
            virtual void PushUpdateComponent();
            virtual void PushDeupdateComponent();
    };

}

#endif
