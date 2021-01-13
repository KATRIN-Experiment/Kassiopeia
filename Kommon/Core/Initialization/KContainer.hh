#ifndef Kommon_KContainer_hh_
#define Kommon_KContainer_hh_

#include "KNamed.h"

namespace katrin
{

class KContainer : public KNamed
{
  private:
    class KHolder
    {
      public:
        KHolder();
        virtual ~KHolder();

      public:
        virtual void Type() = 0;
        virtual void Clear() = 0;
    };

    template<class XType> class KHolderPrototype : public KHolder
    {
      public:
        KHolderPrototype(XType* anObject);
        ~KHolderPrototype() override;

      public:
        void Type() override;
        void Clear() override;

      private:
        XType* fObject;
    };

  public:
    KContainer();
    ~KContainer() override;

    bool Empty() const;

    template<class XTargetType> void Set(XTargetType* aTarget);

    template<class XTargetType> bool Is();

    template<class XTargetType> XTargetType& AsReference();
    template<class XTargetType> XTargetType* AsPointer();

    std::string& AsString()
    {
        return AsReference<std::string>();
    }

    template<class XTargetType> operator XTargetType();

    template<class XTargetType> void CopyTo(XTargetType& aTarget);

    template<class XTargetType> void CopyTo(void (*aTarget)(XTargetType&));
    template<class XTargetType> void CopyTo(void (*aTarget)(const XTargetType&));

    template<class XObjectType, class XMemberType, class XTargetType>
    void CopyTo(XObjectType* aBearer, void (XMemberType::*aMember)(XTargetType));
    template<class XObjectType, class XMemberType, class XTargetType>
    void CopyTo(XObjectType* aBearer, void (XMemberType::*aMember)(XTargetType&));
    template<class XObjectType, class XMemberType, class XTargetType>
    void CopyTo(XObjectType* aBearer, void (XMemberType::*aMember)(const XTargetType&));

    KContainer* ReleaseToNewContainer();

    template<class XTargetType> void ReleaseTo(XTargetType*& aTarget);

    template<class XTargetType> void ReleaseTo(void (*aTarget)(XTargetType*));
    template<class XTargetType> void ReleaseTo(void (*aTarget)(const XTargetType*));

    template<class XObjectType, class XMemberType, class XTargetType>
    void ReleaseTo(XObjectType* aBearer, void (XMemberType::*aMember)(XTargetType*));
    template<class XObjectType, class XMemberType, class XTargetType>
    void ReleaseTo(XObjectType* aBearer, void (XMemberType::*aMember)(const XTargetType*));

  private:
    KHolder* fHolder;
};

inline KContainer::KHolder::KHolder() = default;
inline KContainer::KHolder::~KHolder() = default;

template<class XType> inline KContainer::KHolderPrototype<XType>::KHolderPrototype(XType* anObject) : fObject(anObject)
{}
template<class XType> inline KContainer::KHolderPrototype<XType>::~KHolderPrototype()
{
    if (fObject != nullptr) {
        delete fObject;
        fObject = nullptr;
    }
}
template<class XType> inline void KContainer::KHolderPrototype<XType>::Type()
{
    throw fObject;
    return;
}
template<class XType> inline void KContainer::KHolderPrototype<XType>::Clear()
{
    fObject = nullptr;
    return;
}

template<class XTargetType> inline void KContainer::Set(XTargetType* aType)
{
    if (fHolder != nullptr) {
        delete fHolder;
        fHolder = nullptr;
    }
    auto* tTypedHolder = new KHolderPrototype<XTargetType>(aType);
    fHolder = tTypedHolder;
}

template<class XTargetType> inline bool KContainer::Is()
{
    try {
        fHolder->Type();
    }
    catch (XTargetType*) {
        return true;
    }
    catch (...) {
        return false;
    }
    return false;
}

template<class XTargetType> inline XTargetType& KContainer::AsReference()
{
    try {
        fHolder->Type();
    }
    catch (XTargetType* tObject) {
        return *tObject;
    }
    catch (...) {
    }

    XTargetType* tTarget = nullptr;
    return *tTarget;  // NOLINT: FIXME: need to return empty reference here
}
template<class XTargetType> inline XTargetType* KContainer::AsPointer()
{
    try {
        fHolder->Type();
    }
    catch (XTargetType* tObject) {
        return tObject;
    }
    catch (...) {
    }

    XTargetType* tTarget = nullptr;
    return tTarget;
}

template<class XTargetType> inline KContainer::operator XTargetType()
{
    return AsReference<XTargetType>();
}

template<class XTargetType> inline void KContainer::CopyTo(XTargetType& aTarget)
{
    try {
        fHolder->Type();
    }
    catch (XTargetType* tObject) {
        aTarget = *tObject;
        return;
    }
    catch (...) {
        return;
    }
}

template<class XTargetType> inline void KContainer::CopyTo(void (*aTarget)(XTargetType&))
{
    try {
        fHolder->Type();
    }
    catch (XTargetType* tObject) {
        (*aTarget)(*tObject);
        return;
    }
    catch (...) {
        return;
    }
}
template<class XTargetType> inline void KContainer::CopyTo(void (*aTarget)(const XTargetType&))
{
    try {
        fHolder->Type();
    }
    catch (XTargetType* tObject) {
        (*aTarget)(*tObject);
        return;
    }
    catch (...) {
        return;
    }
}

template<class XObjectType, class XMemberType, class XTargetType>
inline void KContainer::CopyTo(XObjectType* aBearer, void (XMemberType::*aMember)(XTargetType))
{
    try {
        fHolder->Type();
    }
    catch (XTargetType* tObject) {
        (aBearer->*aMember)(*tObject);
        return;
    }
    catch (...) {
        return;
    }
}
template<class XObjectType, class XMemberType, class XTargetType>
inline void KContainer::CopyTo(XObjectType* aBearer, void (XMemberType::*aMember)(XTargetType&))
{
    try {
        fHolder->Type();
    }
    catch (XTargetType* tObject) {
        (aBearer->*aMember)(*tObject);
        return;
    }
    catch (...) {
        return;
    }
}
template<class XObjectType, class XMemberType, class XTargetType>
inline void KContainer::CopyTo(XObjectType* aBearer, void (XMemberType::*aMember)(const XTargetType&))
{
    try {
        fHolder->Type();
    }
    catch (XTargetType* tObject) {
        (aBearer->*aMember)(*tObject);
        return;
    }
    catch (...) {
        return;
    }
}

inline KContainer* KContainer::ReleaseToNewContainer()
{
    auto* newContainer = new KContainer();
    newContainer->fHolder = fHolder;
    fHolder = nullptr;
    return newContainer;
}


template<class XTargetType> inline void KContainer::ReleaseTo(XTargetType*& aTarget)
{
    try {
        fHolder->Type();
    }
    catch (XTargetType* tObject) {
        aTarget = tObject;
        fHolder->Clear();
        return;
    }
    catch (...) {
        return;
    }
}

template<class XTargetType> inline void KContainer::ReleaseTo(void (*aTarget)(XTargetType*))
{
    try {
        fHolder->Type();
    }
    catch (XTargetType* tObject) {
        (*aTarget)(tObject);
        fHolder->Clear();
        return;
    }
    catch (...) {
        return;
    }
}
template<class XTargetType> inline void KContainer::ReleaseTo(void (*aTarget)(const XTargetType*))
{
    try {
        fHolder->Type();
    }
    catch (XTargetType* tObject) {
        (*aTarget)(tObject);
        fHolder->Clear();
        return;
    }
    catch (...) {
        return;
    }
}

template<class XObjectType, class XMemberType, class XTargetType>
inline void KContainer::ReleaseTo(XObjectType* aBearer, void (XMemberType::*aMember)(XTargetType*))
{
    try {
        fHolder->Type();
    }
    catch (XTargetType* tObject) {
        (aBearer->*aMember)(tObject);
        fHolder->Clear();
        return;
    }
    catch (...) {
        return;
    }
}
template<class XObjectType, class XMemberType, class XTargetType>
inline void KContainer::ReleaseTo(XObjectType* aBearer, void (XMemberType::*aMember)(const XTargetType*))
{
    try {
        fHolder->Type();
    }
    catch (XTargetType* tObject) {
        (aBearer->*aMember)(tObject);
        fHolder->Clear();
        return;
    }
    catch (...) {
        return;
    }
}

}  // namespace katrin

#endif
