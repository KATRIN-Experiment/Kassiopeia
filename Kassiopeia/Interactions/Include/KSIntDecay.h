#ifndef Kassiopeia_KSIntDecay_h_
#define Kassiopeia_KSIntDecay_h_

#include "KSSpaceInteraction.h"
#include "KSIntDecayCalculator.h"

#include <vector>
using std::vector;

namespace Kassiopeia
{

    class KSIntDecayCalculator;

    class KSIntDecay :
        public KSComponentTemplate< KSIntDecay, KSSpaceInteraction >
    {
        public:
            KSIntDecay();
            KSIntDecay( const KSIntDecay& aCopy );
            KSIntDecay* Clone() const;
            virtual ~KSIntDecay();

        public:
            vector<double> CalculateLifetimes(
                    const KSParticle& aTrajectoryInitialParticle
            );

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

            void AddCalculator( KSIntDecayCalculator* const aScatteringCalculator );
            void RemoveCalculator( KSIntDecayCalculator* const aScatteringCalculator );

            void SetEnhancement( double anEnhancement );

        private:
            bool fSplit;
            KSIntDecayCalculator* fCalculator;
            vector< KSIntDecayCalculator* > fCalculators;
            vector< double > fLifeTimes;

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
