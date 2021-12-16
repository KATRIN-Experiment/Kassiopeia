#ifndef KTAGGED_H_
#define KTAGGED_H_

#include "KNamed.h"

#include <ostream>
#include <set>
#include <string>

namespace katrin
{

typedef std::string KTag;
using KTagSet = std::set<std::string>;
using KTagSetIt = KTagSet::iterator;
using KTagSetCIt = KTagSet::const_iterator;

class KTagged : public KNamed
{
  public:
    KTagged();
    KTagged(const KTagged& aCopy);    
    ~KTagged() override;

    inline KTagged& operator=(const KTagged& other);

    bool Empty() const;
    bool HasTag(const KTag& aTag) const;
    bool HasTags(const KTagSet& aTagSet) const;
    bool HasAllTags(const KTagSet& aTagSet) const;
    bool HasTagsFrom(const KTagged* aTagged) const;

    const KTagSet& GetTags() const;
    void SetTag(const KTag& aTag);
    void SetTags(const KTagSet& aTagSet);
    void SetTagsFrom(const KTagged* aTagged);

    void AddTag(const KTag& aTag);
    void AddTags(const KTagSet& aTagSet);
    void AddTagsFrom(const KTagged* aTagged);

    void RemoveTag(const KTag& aTag);
    void RemoveTags(const KTagSet& aTagSet);
    void RemoveTagsFrom(const KTagged* aTagged);

  protected:
    KTagSet fTags;

  public:
    static void OpenTag(const KTag& aTag);
    static void CloseTag(const KTag& aTag);

    static KTagSet sOpenTags;
};

inline KTagged& KTagged::operator=(const KTagged& other)
{
    SetName(other.GetName());
    SetTagsFrom(&other);
    return *this;
}

inline std::ostream& operator<<(std::ostream& aStream, const KTagged& aTagged)
{
    aStream << "<" << aTagged.GetName() << "> with tags <";
    for (auto tIter = aTagged.GetTags().begin(); tIter != aTagged.GetTags().end(); tIter++) {
        if (tIter != aTagged.GetTags().begin()) {
            aStream << " ";
        }
        aStream << *tIter;
    }
    aStream << ">";
    return aStream;
}
}  // namespace katrin


#endif
