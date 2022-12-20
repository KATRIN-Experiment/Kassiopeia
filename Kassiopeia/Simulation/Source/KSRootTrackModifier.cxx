#include "KSRootTrackModifier.h"

#include "KSException.h"
#include "KSModifiersMessage.h"

namespace Kassiopeia
{
KSRootTrackModifier::KSRootTrackModifier() : fModifiers(128), fModifier(nullptr), fTrack(nullptr) {}

KSRootTrackModifier::KSRootTrackModifier(const KSRootTrackModifier& aCopy) :
    KSComponent(aCopy),
    fModifiers(aCopy.fModifiers),
    fModifier(aCopy.fModifier),
    fTrack(aCopy.fTrack)
{}

KSRootTrackModifier* KSRootTrackModifier::Clone() const
{
    return new KSRootTrackModifier(*this);
}

KSRootTrackModifier::~KSRootTrackModifier() = default;

void KSRootTrackModifier::AddModifier(KSTrackModifier* aModifier)
{
    fModifiers.AddElement(aModifier);
    return;
}

void KSRootTrackModifier::RemoveModifier(KSTrackModifier* aModifier)
{
    fModifiers.RemoveElement(aModifier);
    return;
}

void KSRootTrackModifier::SetTrack(KSTrack* aTrack)
{
    fTrack = aTrack;
    return;
}

void KSRootTrackModifier::PushUpdateComponent()
{
    for (int tIndex = 0; tIndex < fModifiers.End(); tIndex++) {
        fModifiers.ElementAt(tIndex)->PushUpdate();
    }
}

void KSRootTrackModifier::PushDeupdateComponent()
{
    for (int tIndex = 0; tIndex < fModifiers.End(); tIndex++) {
        fModifiers.ElementAt(tIndex)->PushDeupdate();
    }
}

bool KSRootTrackModifier::ExecutePreTrackModification()
{
    return ExecutePreTrackModification(*fTrack);
}

bool KSRootTrackModifier::ExecutePostTrackModification()
{
    return ExecutePostTrackModification(*fTrack);
}

bool KSRootTrackModifier::ExecutePreTrackModification(KSTrack& aTrack)
{
    bool hasChangedState = false;
    try {
        for (int tIndex = 0; tIndex < fModifiers.End(); tIndex++) {
            modmsg_debug("<" << GetName() << "> executing pre-track modification <" << fModifiers.ElementAt(tIndex)->GetName() << "> in track " << aTrack.GetTrackId() << eom);
            bool changed = fModifiers.ElementAt(tIndex)->ExecutePreTrackModification(aTrack);
            if (changed)
                hasChangedState = true;
        }
    }
    catch (KSException const& e) {
        throw KSModifierError().Nest(e) << "Failed to modify track <" << aTrack.TrackId() << ">.";
    }
    return hasChangedState;
}

bool KSRootTrackModifier::ExecutePostTrackModification(KSTrack& aTrack)
{
    bool hasChangedState = false;
    try {
        for (int tIndex = 0; tIndex < fModifiers.End(); tIndex++) {
            modmsg_debug("<" << GetName() << "> executing post-track modification <" << fModifiers.ElementAt(tIndex)->GetName() << "> in track " << aTrack.GetTrackId() << eom);
            bool changed = fModifiers.ElementAt(tIndex)->ExecutePostTrackModification(aTrack);
            if (changed)
                hasChangedState = true;
        }
    }
    catch (KSException const& e) {
        throw KSModifierError().Nest(e) << "Failed to modify track <" << aTrack.TrackId() << ">.";
    }
    return hasChangedState;
}


STATICINT sKSRootModifierDict = KSDictionary<KSRootTrackModifier>::AddCommand(
    &KSRootTrackModifier::AddModifier, &KSRootTrackModifier::RemoveModifier, "add_modifier", "remove_modifier");

}  // namespace Kassiopeia
