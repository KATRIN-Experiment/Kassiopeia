#ifndef Kassiopeia_KSObject_h_
#define Kassiopeia_KSObject_h_

#include "KSObjectsMessage.h"
#include "KTagged.h"
#include <memory>

namespace Kassiopeia
{

#if defined(__GNUC__) || defined(__clang__)
// GCC 15 can report a false -Wuninitialized path when this template is inlined into
// component constructors; keep Set(...) out-of-line at call sites to avoid that path.
// Found with Set<...>(this) in the constructor of KSMagneticField, where "this" being
// potentially not fully initialized caused the above issue.
#define KSOBJECT_NOINLINE __attribute__((noinline))
#else
#define KSOBJECT_NOINLINE
#endif

class KSObject : public katrin::KTagged
{
  public:
    KSObject();
    KSObject(const KSObject& aCopy);
    ~KSObject() override;

  public:
    virtual KSObject* Clone() const = 0;

  public:
    template<class XType> bool Is();

    template<class XType> bool Is() const;

    template<class XType> XType* As();

    template<class XType> const XType* As() const;

  protected:
    template<class XType> KSOBJECT_NOINLINE void Set(XType*);

  private:
    class KSHolder
    {
      public:
        KSHolder();
        virtual ~KSHolder();

      public:
        virtual void Type() = 0;
    };

    template<class XType> class KSHolderTemplate : public KSHolder
    {
      public:
        KSHolderTemplate(XType* anObject);
        ~KSHolderTemplate() override;

      public:
        void Type() override;

      private:
        XType* fObject;
    };

    mutable std::unique_ptr<KSHolder> fHolder = nullptr;
};

inline KSObject::KSHolder::KSHolder() = default;
inline KSObject::KSHolder::~KSHolder() = default;

template<class XType> inline KSObject::KSHolderTemplate<XType>::KSHolderTemplate(XType* anObject) : fObject(anObject) {}
template<class XType> inline KSObject::KSHolderTemplate<XType>::~KSHolderTemplate() = default;
template<class XType> inline void KSObject::KSHolderTemplate<XType>::Type()
{
    throw fObject;
    return;
}

template<class XType> inline bool KSObject::Is()
{
    if(!fHolder) {
        return false;
    }
    try {
        fHolder->Type();
    }
    catch (XType* tObject) {
        return true;
    }
    catch (...) {
        return false;
    }
    return false;
}
template<> inline bool KSObject::Is<KSObject>()
{
    return true;
}

template<class XType> inline bool KSObject::Is() const
{
    if(!fHolder) {
        return false;
    }
    try {
        fHolder->Type();
    }
    catch (XType* tObject) {
        return true;
    }
    catch (...) {
        return false;
    }
    return false;
}
template<> inline bool KSObject::Is<KSObject>() const
{
    return true;
}

template<class XType> inline XType* KSObject::As()
{
    if(!fHolder) {
        return nullptr;
    }
    try {
        fHolder->Type();
    }
    catch (XType* tObject) {
        return tObject;
    }
    catch (...) {
        return nullptr;
    }
    return nullptr;
}
template<> inline KSObject* KSObject::As<KSObject>()
{
    return this;
}

template<class XType> inline const XType* KSObject::As() const
{
    if(!fHolder) {
        return nullptr;
    }
    try {
        fHolder->Type();
    }
    catch (XType* tObject) {
        return tObject;
    }
    catch (...) {
        return NULL;
    }
    return NULL;
}
template<> inline const KSObject* KSObject::As<KSObject>() const
{
    return this;
}

template<class XType> KSOBJECT_NOINLINE void KSObject::Set(XType* anObject)
{
    auto* tHolder = new KSHolderTemplate<XType>(anObject);
    fHolder = std::unique_ptr<KSHolder>(tHolder);
    return;
}

}  // namespace Kassiopeia

#undef KSOBJECT_NOINLINE

#endif
