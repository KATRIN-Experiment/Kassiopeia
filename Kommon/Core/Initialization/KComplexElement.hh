#ifndef Kommon_KComplexElement_hh_
#define Kommon_KComplexElement_hh_

#ifndef STATICINT
#define STATICINT static const int __attribute__((__unused__))
#endif

#include "KAttribute.hh"
#include "KAttributeBase.hh"
#include "KElementBase.hh"
#include "KSimpleElement.hh"

#include <string>
using std::string;

namespace katrin
{

template<class XType> class KComplexElement : public KElementBase
{
  public:
    KComplexElement(KElementBase* aParentElement = nullptr);
    ~KComplexElement() override;

    bool Begin() override;
    bool AddAttribute(KContainer* aToken) override;
    bool Body() override;
    bool AddElement(KContainer* anElement) override;
    bool SetValue(KToken* aValue) override;
    bool End() override;

    static KElementBase* Create(KElementBase* aParentElement);
    template<class XAttributeType> static int Attribute(const std::string& aName);
    template<class XElementType> static int SimpleElement(const std::string& aName);
    template<class XElementType> static int ComplexElement(const std::string& aName);

  protected:
    XType* fObject;
    static KAttributeMap* sAttributes;
    static KElementMap* sElements;
};

template<class XType> KComplexElement<XType>::KComplexElement(KElementBase* aParentElement) : fObject(nullptr)
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
template<class XType> KComplexElement<XType>::~KComplexElement() {}

template<class XType> bool KComplexElement<XType>::Begin()
{
    fObject = new XType();
    Set(fObject);
    return true;
}
template<class XType> bool KComplexElement<XType>::AddAttribute(KContainer*)
{
    return true;
}
template<class XType> bool KComplexElement<XType>::Body()
{
    return true;
}
template<class XType> bool KComplexElement<XType>::AddElement(KContainer*)
{
    return true;
}
template<class XType> bool KComplexElement<XType>::SetValue(KToken*)
{
    return true;
}
template<class XType> bool KComplexElement<XType>::End()
{
    return true;
}

template<class XType> KElementBase* KComplexElement<XType>::Create(KElementBase* aParentElement)
{
    return new KComplexElement<XType>(aParentElement);
}

template<class XType> KAttributeMap* KComplexElement<XType>::sAttributes = nullptr;
template<class XType> template<class XAttributeType> int KComplexElement<XType>::Attribute(const std::string& aName)
{
    if (sAttributes == nullptr) {
        sAttributes = new KAttributeMap();
    }
    KComplexElement<XType>::sAttributes->insert(KAttributeEntry(aName, &KAttribute<XAttributeType>::Create));
    return 0;
}
template<class XType> KElementMap* KComplexElement<XType>::sElements = nullptr;
template<class XType> template<class XElementType> int KComplexElement<XType>::SimpleElement(const std::string& aName)
{
    if (sElements == nullptr) {
        sElements = new KElementMap();
    }
    KComplexElement<XType>::sElements->insert(KElementEntry(aName, &KSimpleElement<XElementType>::Create));
    return 0;
}
template<class XType> template<class XElementType> int KComplexElement<XType>::ComplexElement(const std::string& aName)
{
    if (sElements == nullptr) {
        sElements = new KElementMap();
    }
    KComplexElement<XType>::sElements->insert(KElementEntry(aName, &KComplexElement<XElementType>::Create));
    return 0;
}
}  // namespace katrin

/*
 * handlers for automatic input conversion into common sequence types
 */
namespace std
{

template<class T> std::istream& operator>>(std::istream& stream, std::vector<T>& data)
{
    while (!stream.eof()) {
        std::string str;
        stream >> str;
        data.push_back(str);
    }
    return stream;
}

template<class T> std::istream& operator>>(std::istream& stream, std::list<T>& data)
{
    while (!stream.eof()) {
        std::string str;
        stream >> str;
        data.push_back(str);
    }
    return stream;
}

template<class T> std::istream& operator>>(std::istream& stream, std::set<T>& data)
{
    while (!stream.eof()) {
        std::string str;
        stream >> str;
        data.insert(str);
    }
    return stream;
}

}  // namespace std

#endif
