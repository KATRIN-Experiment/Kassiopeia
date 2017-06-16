#include "KSRootTrackModifier.h"
#include "KSModifiersMessage.h"

namespace Kassiopeia
{
    KSRootTrackModifier::KSRootTrackModifier() :
        fModifiers(128),
        fModifier( NULL ),
        fTrack(NULL)
    {
    }

    KSRootTrackModifier::KSRootTrackModifier(const KSRootTrackModifier &aCopy) : KSComponent(),
        fModifiers( aCopy.fModifiers ),
        fModifier( aCopy.fModifier ),
        fTrack( aCopy.fTrack )
    {
    }

    KSRootTrackModifier* KSRootTrackModifier::Clone() const
    {
        return new KSRootTrackModifier( *this );
    }

    KSRootTrackModifier::~KSRootTrackModifier()
    {
    }

    void KSRootTrackModifier::AddModifier(KSTrackModifier *aModifier)
    {
        fModifiers.AddElement( aModifier );
        return;
    }

    void KSRootTrackModifier::RemoveModifier(KSTrackModifier *aModifier)
    {
        fModifiers.RemoveElement( aModifier );
        return;
    }

    void KSRootTrackModifier::SetTrack( KSTrack* aTrack )
    {
        fTrack = aTrack;
        return;
    }

    void KSRootTrackModifier::PushUpdateComponent()
    {
        for( int tIndex = 0; tIndex < fModifiers.End(); tIndex++ )
        {
            fModifiers.ElementAt( tIndex )->PushUpdate();
        }
    }

    void KSRootTrackModifier::PushDeupdateComponent()
    {
        for( int tIndex = 0; tIndex < fModifiers.End(); tIndex++ )
        {
            fModifiers.ElementAt( tIndex )->PushDeupdate();
        }
    }

    bool KSRootTrackModifier::ExecutePreTrackModification()
    {
        return ExecutePreTrackModification( *fTrack );
    }

    bool KSRootTrackModifier::ExecutePostTrackModification()
    {
        return ExecutePostTrackModification( *fTrack );
    }

    bool KSRootTrackModifier::ExecutePreTrackModification(KSTrack& aTrack)
    {
        bool hasChangedState = false;
        for( int tIndex = 0; tIndex < fModifiers.End(); tIndex++ )
        {
            bool changed = fModifiers.ElementAt( tIndex )->ExecutePreTrackModification( aTrack );
            if(changed){hasChangedState = true;};
        }

        return hasChangedState;
    }

    bool KSRootTrackModifier::ExecutePostTrackModification( KSTrack& aTrack )
    {
        bool hasChangedState = false;
        for( int tIndex = 0; tIndex < fModifiers.End(); tIndex++ )
        {
            bool changed = fModifiers.ElementAt( tIndex )->ExecutePostTrackModification( aTrack );
            if(changed){hasChangedState = true;};
        }

        return hasChangedState;
    }


    STATICINT sKSRootModifierDict =
            KSDictionary< KSRootTrackModifier >::AddCommand( &KSRootTrackModifier::AddModifier,
                                                            &KSRootTrackModifier::RemoveModifier,
                                                            "add_modifier",
                                                            "remove_modifier" );

}
