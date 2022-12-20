#include "KSRootEventModifier.h"

#include "KSException.h"
#include "KSModifiersMessage.h"

namespace Kassiopeia
{
KSRootEventModifier::KSRootEventModifier() : fModifiers(128), fModifier(nullptr), fEvent(nullptr) {}

KSRootEventModifier::KSRootEventModifier(const KSRootEventModifier& aCopy) :
    KSComponent(aCopy),
    fModifiers(aCopy.fModifiers),
    fModifier(aCopy.fModifier),
    fEvent(aCopy.fEvent)
{}

KSRootEventModifier* KSRootEventModifier::Clone() const
{
    return new KSRootEventModifier(*this);
}

KSRootEventModifier::~KSRootEventModifier() = default;

void KSRootEventModifier::AddModifier(KSEventModifier* aModifier)
{
    fModifiers.AddElement(aModifier);
    return;
}

void KSRootEventModifier::RemoveModifier(KSEventModifier* aModifier)
{
    fModifiers.RemoveElement(aModifier);
    return;
}

void KSRootEventModifier::SetEvent(KSEvent* anEvent)
{
    fEvent = anEvent;
    return;
}

void KSRootEventModifier::PushUpdateComponent()
{
    for (int tIndex = 0; tIndex < fModifiers.End(); tIndex++) {
        fModifiers.ElementAt(tIndex)->PushUpdate();
    }
}

void KSRootEventModifier::PushDeupdateComponent()
{
    for (int tIndex = 0; tIndex < fModifiers.End(); tIndex++) {
        fModifiers.ElementAt(tIndex)->PushDeupdate();
    }
}

bool KSRootEventModifier::ExecutePreEventModification()
{
    return ExecutePreEventModification(*fEvent);
}

bool KSRootEventModifier::ExecutePostEventModification()
{
    return ExecutePostEventModification(*fEvent);
}

bool KSRootEventModifier::ExecutePreEventModification(KSEvent& anEvent)
{
    bool hasChangedState = false;
    try {
        for (int tIndex = 0; tIndex < fModifiers.End(); tIndex++) {
            modmsg_debug("<" << GetName() << "> executing pre-event modification <" << fModifiers.ElementAt(tIndex)->GetName() << "> in track " << anEvent.GetEventId() << eom);
            bool changed = fModifiers.ElementAt(tIndex)->ExecutePreEventModification(anEvent);
            if (changed)
                hasChangedState = true;
        }
    }
    catch (KSException const& e) {
        throw KSModifierError().Nest(e) << "Failed to modify event <" << anEvent.EventId() << ">.";
    }
    return hasChangedState;
}

bool KSRootEventModifier::ExecutePostEventModification(KSEvent& anEvent)
{
    bool hasChangedState = false;
    try {
        for (int tIndex = 0; tIndex < fModifiers.End(); tIndex++) {
            modmsg_debug("<" << GetName() << "> executing post-event modification <" << fModifiers.ElementAt(tIndex)->GetName() << "> in track " << anEvent.GetEventId() << eom);
            bool changed = fModifiers.ElementAt(tIndex)->ExecutePostEventModification(anEvent);
            if (changed)
                hasChangedState = true;
        }
    }
    catch (KSException const& e) {
        throw KSModifierError().Nest(e) << "Failed to modify event <" << anEvent.EventId() << ">.";
    }
    return hasChangedState;
}


STATICINT sKSRootModifierDict = KSDictionary<KSRootEventModifier>::AddCommand(
    &KSRootEventModifier::AddModifier, &KSRootEventModifier::RemoveModifier, "add_modifier", "remove_modifier");

}  // namespace Kassiopeia
