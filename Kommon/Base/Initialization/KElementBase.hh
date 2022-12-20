#ifndef Kommon_KElementBase_hh_
#define Kommon_KElementBase_hh_

#include "KContainer.hh"
#include "KProcessor.hh"

#include <map>
#include <algorithm>

namespace katrin
{
class KAttributeBase;
class KElementBase;

using KAttributeMap = std::map<std::string, KAttributeBase* (*) (KElementBase*)>;
using KAttributeEntry = KAttributeMap::value_type;
using KAttributeIt = KAttributeMap::iterator;
using KAttributeCIt = KAttributeMap::const_iterator;
using KAttributeList = std::vector<KAttributeMap::key_type>;

using KElementMap = std::map<std::string, KElementBase* (*) (KElementBase*)>;
using KElementEntry = KElementMap::value_type;
using KElementIt = KElementMap::iterator;
using KElementCIt = KElementMap::const_iterator;
using KElementList = std::vector<KElementMap::key_type>;


class KElementBase : public KContainer, public KProcessor
{
  public:
    KElementBase();
    ~KElementBase() override;

  public:
    void ProcessToken(KBeginElementToken* aToken) override;
    void ProcessToken(KBeginAttributeToken* aToken) override;
    void ProcessToken(KEndAttributeToken* aToken) override;
    void ProcessToken(KMidElementToken* aToken) override;
    void ProcessToken(KElementDataToken* aToken) override;
    void ProcessToken(KEndElementToken* aToken) override;
    void ProcessToken(KErrorToken* aToken) override;

  public:
    virtual bool Begin() = 0;
    virtual bool Body() = 0;
    virtual bool End() = 0;
    virtual bool SetValue(KToken* aValue) = 0;
    virtual bool AddAttribute(KContainer* aToken) = 0;
    virtual bool AddElement(KContainer* anElement) = 0;

  protected:
    virtual KAttributeList GetAttributes() const;
    virtual KElementList GetElements() const;

  protected:
    KElementBase* fParentElement;

    const KAttributeMap* fAttributes;
    KAttributeBase* fChildAttribute;
    unsigned int fAttributeDepth;

    const KElementMap* fElements;
    KElementBase* fChildElement;
    unsigned int fElementDepth;
};

inline KAttributeList KElementBase::GetAttributes() const
{
    KAttributeList keys;
    std::transform(fAttributes->begin(), fAttributes->end(), std::back_inserter(keys), [](const KAttributeEntry& it) {
        return it.first;
    });
    return keys;
}

inline KElementList KElementBase::GetElements() const
{
    KElementList keys;
    std::transform(fElements->begin(), fElements->end(), std::back_inserter(keys), [](const KElementEntry& it) {
        return it.first;
    });
    return keys;
}
}  // namespace katrin

#endif
