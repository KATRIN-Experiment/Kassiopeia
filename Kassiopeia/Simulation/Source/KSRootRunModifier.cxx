#include "KSRootRunModifier.h"

#include "KSException.h"
#include "KSModifiersMessage.h"

namespace Kassiopeia
{
KSRootRunModifier::KSRootRunModifier() : fModifiers(128), fModifier(nullptr), fRun(nullptr) {}

KSRootRunModifier::KSRootRunModifier(const KSRootRunModifier& aCopy) :
    KSComponent(aCopy),
    fModifiers(aCopy.fModifiers),
    fModifier(aCopy.fModifier),
    fRun(aCopy.fRun)
{}

KSRootRunModifier* KSRootRunModifier::Clone() const
{
    return new KSRootRunModifier(*this);
}

KSRootRunModifier::~KSRootRunModifier() = default;

void KSRootRunModifier::AddModifier(KSRunModifier* aModifier)
{
    fModifiers.AddElement(aModifier);
    return;
}

void KSRootRunModifier::RemoveModifier(KSRunModifier* aModifier)
{
    fModifiers.RemoveElement(aModifier);
    return;
}

void KSRootRunModifier::SetRun(KSRun* aRun)
{
    fRun = aRun;
    return;
}

void KSRootRunModifier::PushUpdateComponent()
{
    for (int tIndex = 0; tIndex < fModifiers.End(); tIndex++) {
        fModifiers.ElementAt(tIndex)->PushUpdate();
    }
}

void KSRootRunModifier::PushDeupdateComponent()
{
    for (int tIndex = 0; tIndex < fModifiers.End(); tIndex++) {
        fModifiers.ElementAt(tIndex)->PushDeupdate();
    }
}

bool KSRootRunModifier::ExecutePreRunModification()
{
    return ExecutePreRunModification(*fRun);
}

bool KSRootRunModifier::ExecutePostRunModification()
{
    return ExecutePostRunModification(*fRun);
}

bool KSRootRunModifier::ExecutePreRunModification(KSRun& aRun)
{
    bool hasChangedState = false;
    try {
        for (int tIndex = 0; tIndex < fModifiers.End(); tIndex++) {
            modmsg_debug("<" << GetName() << "> executing pre-run modification <" << fModifiers.ElementAt(tIndex)->GetName() << "> in run " << aRun.GetRunId() << eom);
            bool changed = fModifiers.ElementAt(tIndex)->ExecutePreRunModification(aRun);
            if (changed)
                hasChangedState = true;
        }
    }
    catch (KSException const& e) {
        throw KSModifierError().Nest(e) << "Failed to modify run <" << aRun.RunId() << ">.";
    }
    return hasChangedState;
}

bool KSRootRunModifier::ExecutePostRunModification(KSRun& aRun)
{
    bool hasChangedState = false;
    try {
        for (int tIndex = 0; tIndex < fModifiers.End(); tIndex++) {
            modmsg_debug("<" << GetName() << "> executing post-run modification <" << fModifiers.ElementAt(tIndex)->GetName() << "> in run " << aRun.GetRunId() << eom);
            bool changed = fModifiers.ElementAt(tIndex)->ExecutePostRunModification(aRun);
            if (changed)
                hasChangedState = true;
        }
    }
    catch (KSException const& e) {
        throw KSModifierError().Nest(e) << "Failed to modify run <" << aRun.RunId() << ">.";
    }
    return hasChangedState;
}


STATICINT sKSRootModifierDict = KSDictionary<KSRootRunModifier>::AddCommand(
    &KSRootRunModifier::AddModifier, &KSRootRunModifier::RemoveModifier, "add_modifier", "remove_modifier");

}  // namespace Kassiopeia
