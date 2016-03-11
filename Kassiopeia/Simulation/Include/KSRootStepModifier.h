#ifndef Kassiopeia_KSRootStepModifier_h_
#define Kassiopeia_KSRootStepModifier_h_

#include "KSStepModifier.h"
#include "KSStep.h"
#include "KSList.h"
#include "KSParticle.h"

namespace Kassiopeia
{

    class KSTrack;

    class KSRootStepModifier :
            public KSComponentTemplate< KSRootStepModifier, KSStepModifier >
    {
    public:
        KSRootStepModifier();
        KSRootStepModifier( const KSRootStepModifier& aCopy );
        KSRootStepModifier* Clone() const;
        virtual ~KSRootStepModifier();

        //**********
        // modifier
        //**********

    public:
        bool ExecutePreStepModification( KSParticle& anInitialParticle, KSParticleQueue& aQueue );
        bool ExecutePostStepModifcation( KSParticle& anInitialParticle, KSParticle& aFinalParticle, KSParticleQueue& aQueue );

        //***********
        //composition
        //***********

    public:
        void AddModifier( KSStepModifier* aModifier );
        void RemoveModifier( KSStepModifier* aModifier );

    private:
        KSList< KSStepModifier > fModifiers;
        KSStepModifier* fModifier;

        //******
        //action
        //******

    public:
        void SetStep( KSStep* aStep );

        bool ExecutePreStepModification();
        bool ExecutePostStepModifcation();

        virtual void PushUpdateComponent();
        virtual void PushDeupdateComponent();

    private:
        KSStep* fStep;
        const KSParticle* fInitialParticle;
        KSParticle* fModifierParticle;
        KSParticle* fFinalParticle;
        KSParticleQueue* fParticleQueue;
    };


}

#endif
