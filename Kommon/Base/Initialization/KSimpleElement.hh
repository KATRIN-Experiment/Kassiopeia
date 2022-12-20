#ifndef Kommon_KSimpleElement_hh_
#define Kommon_KSimpleElement_hh_

#include "KElementBase.hh"

namespace katrin
{

template<class XType> class KSimpleElement : public KElementBase
{
  public:
    KSimpleElement(KElementBase* aParentElement = nullptr);
    ~KSimpleElement() override;

    bool Begin() override;
    bool AddAttribute(KContainer* aToken) override;
    bool Body() override;
    bool AddElement(KContainer* anElement) override;
    bool SetValue(KToken* aToken) override;
    bool End() override;

    static KElementBase* Create(KElementBase* aParentElement);

  protected:
    XType* fObject;
    static KAttributeMap* sAttributes;
    static KElementMap* sElements;
};

template<class XType> KSimpleElement<XType>::KSimpleElement(KElementBase* aParentElement) : fObject(nullptr)
{
    fParentElement = aParentElement;

    if (sElements == nullptr) {
        sElements = new KElementMap();
    }
    fElements = sElements;

    if (sAttributes == nullptr) {
        sAttributes = new KAttributeMap();
    }
    fAttributes = sAttributes;
}
template<class XType> KSimpleElement<XType>::~KSimpleElement() = default;

template<class XType> bool KSimpleElement<XType>::Begin()
{
    fObject = new XType();
    Set(fObject);
    return true;
}
template<class XType> bool KSimpleElement<XType>::AddAttribute(KContainer*)
{
    return false;
}
template<class XType> bool KSimpleElement<XType>::Body()
{
    return true;
}
template<class XType> bool KSimpleElement<XType>::AddElement(KContainer*)
{
    return false;
}
template<class XType> bool KSimpleElement<XType>::SetValue(KToken* aToken)
{
    (*fObject) = aToken->GetValue<XType>();
    return true;
}
template<class XType> bool KSimpleElement<XType>::End()
{
    return true;
}

template<class XType> KElementBase* KSimpleElement<XType>::Create(KElementBase* aParentElement)
{
    return new KSimpleElement<XType>(aParentElement);
}

template<class XType> KAttributeMap* KSimpleElement<XType>::sAttributes = nullptr;

template<class XType> KElementMap* KSimpleElement<XType>::sElements = nullptr;
}  // namespace katrin

#endif
