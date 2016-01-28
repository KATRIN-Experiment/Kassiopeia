#ifndef Kassiopeia_KSModDynamicEnhancement_h_
#define Kassiopeia_KSModDynamicEnhancement_h_

#include "KField.h"
#include "KSStepModifier.h"
#include "KSIntScattering.h"
#include "KSTrajTermSynchrotron.h"
#include "KSComponentTemplate.h"

namespace Kassiopeia
{

    class KSModDynamicEnhancement :
            public KSComponentTemplate< KSModDynamicEnhancement, KSStepModifier >
    {
        public:
            KSModDynamicEnhancement();
            KSModDynamicEnhancement( const KSModDynamicEnhancement& aCopy);
            KSModDynamicEnhancement* Clone() const;
            virtual ~KSModDynamicEnhancement();

        public:
            bool ExecutePreStepModification( KSParticle& anInitialParticle, KSParticleQueue& aQueue );
            bool ExecutePostStepModifcation( KSParticle& anInitialParticle, KSParticle& aFinalParticle, KSParticleQueue& aQueue );

        public:
            K_GET(double, Enhancement )
            K_SET_GET(double, StaticEnhancement )
            K_SET_GET(bool, Dynamic)
            K_SET_GET(double, ReferenceCrossSectionEnergy)

        public:
            void SetScattering( KSIntScattering* aScattering );
            void SetSynchrotron( KSTrajTermSynchrotron* aSynchrotron );

        private:
            KSIntScattering* fScattering;
            KSTrajTermSynchrotron* fSynchrotron;
            double fReferenceCrossSection;

        private:
            void InitializeComponent();
            void DeinitializeComponent();

        protected:
            virtual void PullDeupdateComponent();
            virtual void PushDeupdateComponent();
    };
}

#endif
