#ifndef Kassiopeia_KSRootEventModifier_h_
#define Kassiopeia_KSRootEventModifier_h_

#include "KSEventModifier.h"
#include "KSEvent.h"
#include "KSList.h"
#include "KSParticle.h"

namespace Kassiopeia
{

    class KSTrack;

    class KSRootEventModifier :
            public KSComponentTemplate< KSRootEventModifier, KSEventModifier >
    {
    public:
        KSRootEventModifier();
        KSRootEventModifier( const KSRootEventModifier& aCopy );
        KSRootEventModifier* Clone() const;
        virtual ~KSRootEventModifier();

        //**********
        // modifier
        //**********

    public:
        //bool ExecutePreEventModification( KSParticle& anInitialParticle, KSParticleQueue& aQueue );
        //bool ExecutePostEventModifcation( KSParticle& anInitialParticle, KSParticle& aFinalParticle, KSParticleQueue& aQueue );

        //***********
        //composition
        //***********

    public:
        void AddModifier( KSEventModifier* aModifier );
        void RemoveModifier( KSEventModifier* aModifier );

    private:
        KSList< KSEventModifier > fModifiers;
        KSEventModifier* fModifier;

        //******
        //action
        //******

    public:
        void SetEvent( KSEvent* aEvent );

        bool ExecutePreEventModification();
        bool ExecutePostEventModifcation();

        virtual void PushUpdateComponent();
        virtual void PushDeupdateComponent();

    private:
        KSEvent* fEvent;
    };


}

#endif
