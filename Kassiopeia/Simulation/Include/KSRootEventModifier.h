#ifndef Kassiopeia_KSRootEventModifier_h_
#define Kassiopeia_KSRootEventModifier_h_

#include "KSEventModifier.h"
#include "KSParticle.h"
#include "KSStep.h"
#include "KSEvent.h"
#include "KSTrack.h"
#include "KSList.h"

namespace Kassiopeia
{

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

        bool ExecutePreEventModification( KSEvent& anEvent );
        bool ExecutePostEventModification( KSEvent& anEvent );

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

        void SetEvent( KSEvent* anEvent );

        bool ExecutePreEventModification();
        bool ExecutePostEventModification();

        virtual void PushUpdateComponent();
        virtual void PushDeupdateComponent();

    private:
        KSEvent* fEvent;
    };


}

#endif
