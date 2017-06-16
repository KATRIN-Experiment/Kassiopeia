#ifndef Kassiopeia_KSRootTrackModifier_h_
#define Kassiopeia_KSRootTrackModifier_h_

#include "KSTrackModifier.h"
#include "KSParticle.h"
#include "KSStep.h"
#include "KSTrack.h"
#include "KSList.h"

namespace Kassiopeia
{

    class KSRootTrackModifier :
            public KSComponentTemplate< KSRootTrackModifier, KSTrackModifier >
    {
    public:
        KSRootTrackModifier();
        KSRootTrackModifier( const KSRootTrackModifier& aCopy );
        KSRootTrackModifier* Clone() const;
        virtual ~KSRootTrackModifier();

        //**********
        // modifier
        //**********

    public:
        bool ExecutePreTrackModification( KSTrack& aTrack );
        bool ExecutePostTrackModification( KSTrack& aTrack );

        //***********
        //composition
        //***********

    public:
        void AddModifier( KSTrackModifier* aModifier );
        void RemoveModifier( KSTrackModifier* aModifier );

    private:
        KSList< KSTrackModifier > fModifiers;
        KSTrackModifier* fModifier;

        //******
        //action
        //******

    public:
        void SetTrack( KSTrack* aTrack );

        bool ExecutePreTrackModification();
        bool ExecutePostTrackModification();

        virtual void PushUpdateComponent();
        virtual void PushDeupdateComponent();

    private:

        KSTrack* fTrack;
    };


}

#endif
