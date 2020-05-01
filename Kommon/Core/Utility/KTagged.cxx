#include "KTagged.h"

namespace katrin
{

KTagged::KTagged() : KNamed(), fTags(sOpenTags) {}
KTagged::KTagged(const KTagged& aCopy) : KNamed(aCopy), fTags(aCopy.fTags) {}
KTagged::~KTagged() {}

bool KTagged::Empty() const
{
    if (fTags.size() == 0) {
        return true;
    }
    return false;
}
bool KTagged::HasTag(const KTag& aTag) const
{
    KTagSetCIt tTagIter;
    tTagIter = fTags.find(aTag);
    if (tTagIter != fTags.end()) {
        return true;
    }
    return false;
}
bool KTagged::HasTags(const KTagSet& aTagSet) const
{
    KTagSetCIt tExtTagIt;
    KTagSetCIt tIntTagIt;
    for (tExtTagIt = aTagSet.begin(); tExtTagIt != aTagSet.end(); tExtTagIt++) {
        tIntTagIt = fTags.find(*tExtTagIt);
        if (tIntTagIt != fTags.end()) {
            return true;
        }
    }
    return false;
}
bool KTagged::HasAllTags(const KTagSet& aTagSet) const
{
    KTagSetCIt tExtTagIt;
    KTagSetCIt tIntTagIt;
    for (tExtTagIt = aTagSet.begin(); tExtTagIt != aTagSet.end(); tExtTagIt++) {
        tIntTagIt = fTags.find(*tExtTagIt);
        if (tIntTagIt == fTags.end()) {
            return false;
        }
    }
    return true;
}
bool KTagged::HasTagsFrom(const KTagged* aTagged) const
{
    return HasTags(aTagged->GetTags());
}

const KTagSet& KTagged::GetTags() const
{
    return fTags;
}
void KTagged::SetTag(const KTag& aTag)
{
    fTags.clear();
    fTags.insert(aTag);
    return;
}
void KTagged::SetTags(const KTagSet& aTagSet)
{
    fTags = aTagSet;
    return;
}
void KTagged::SetTagsFrom(const KTagged* aTagged)
{
    return SetTags(aTagged->fTags);
}

void KTagged::AddTag(const KTag& aTag)
{
    fTags.insert(aTag);
    return;
}
void KTagged::AddTags(const KTagSet& aTagSet)
{
    KTagSetCIt tIter;
    for (tIter = aTagSet.begin(); tIter != aTagSet.end(); tIter++) {
        fTags.insert(*tIter);
    }
    return;
}
void KTagged::AddTagsFrom(const KTagged* aTagSet)
{
    return AddTags(aTagSet->fTags);
}

void KTagged::RemoveTag(const KTag& aTag)
{
    KTagSetIt tIntIter;
    tIntIter = fTags.find(aTag);
    if (tIntIter != fTags.end()) {
        fTags.erase(tIntIter);
    }
    return;
}
void KTagged::RemoveTags(const KTagSet& aTagSet)
{
    KTagSetCIt tExtIter;
    KTagSetIt tIntIter;
    for (tExtIter = aTagSet.begin(); tExtIter != aTagSet.end(); tExtIter++) {
        tIntIter = fTags.find(*tExtIter);
        if (tIntIter != fTags.end()) {
            fTags.erase(tIntIter);
        }
    }
    return;
}
void KTagged::RemoveTagsFrom(const KTagged* aTagSet)
{
    return RemoveTags(aTagSet->fTags);
}


KTagSet KTagged::sOpenTags = KTagSet();
void KTagged::OpenTag(const KTag& aTag)
{
    KTagSetIt tTagIter;
    tTagIter = sOpenTags.find(aTag);
    if (tTagIter == sOpenTags.end()) {
        sOpenTags.insert(aTag);
    }
    return;
}
void KTagged::CloseTag(const KTag& aTag)
{
    KTagSetIt tTagIter;
    tTagIter = sOpenTags.find(aTag);
    if (tTagIter != sOpenTags.end()) {
        sOpenTags.erase(tTagIter);
    }
    return;
}
}  // namespace katrin
