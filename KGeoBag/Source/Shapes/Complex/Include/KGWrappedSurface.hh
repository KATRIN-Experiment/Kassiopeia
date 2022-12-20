#ifndef KGWRAPPEDSURFACE_HH_
#define KGWRAPPEDSURFACE_HH_

#include "KGCore.hh"

namespace KGeoBag
{
template<class XObject> class KGWrappedSurface : public KGArea
{
  public:
    class Visitor
    {
      public:
        Visitor() = default;
        virtual ~Visitor() = default;

        virtual void VisitWrappedSurface(KGWrappedSurface<XObject>* aWrappedSurface) = 0;
    };

  public:
    KGWrappedSurface();
    KGWrappedSurface(XObject* anObject);
    KGWrappedSurface(const std::shared_ptr<XObject>& anObject);
    KGWrappedSurface(const KGWrappedSurface& aCopy);
    ~KGWrappedSurface() override;

    static std::string Name()
    {
        return "wrapped_" + XObject::Name() + "_surface";
    }

  public:
    void SetObject(std::shared_ptr<XObject> anObject);
    std::shared_ptr<XObject> GetObject();
    const std::shared_ptr<XObject> GetObject() const;

  public:
    void AreaInitialize() const override;
    void AreaAccept(KGVisitor* aVisitor) override;
    bool AreaAbove(const katrin::KThreeVector& aPoint) const override;
    katrin::KThreeVector AreaPoint(const katrin::KThreeVector& aPoint) const override;
    katrin::KThreeVector AreaNormal(const katrin::KThreeVector& aPoint) const override;

  protected:
    mutable std::shared_ptr<XObject> fObject;
};

template<class XObject> KGWrappedSurface<XObject>::KGWrappedSurface() : KGArea(), fObject(new XObject) {}

template<class XObject> KGWrappedSurface<XObject>::KGWrappedSurface(XObject* anObject) : KGArea(), fObject(anObject) {}

template<class XObject> KGWrappedSurface<XObject>::KGWrappedSurface(const std::shared_ptr<XObject>& anObject) : KGArea()
{
    fObject = anObject;
}

template<class XObject> KGWrappedSurface<XObject>::KGWrappedSurface(const KGWrappedSurface& aCopy) : KGArea(aCopy)
{
    fObject = aCopy.fObject;
}

template<class XObject> KGWrappedSurface<XObject>::~KGWrappedSurface() = default;

template<class XObject> void KGWrappedSurface<XObject>::SetObject(std::shared_ptr<XObject> anObject)
{
    fObject = anObject;
    return;
}

template<class XObject> std::shared_ptr<XObject> KGWrappedSurface<XObject>::GetObject()
{
    return fObject;
}

template<class XObject> const std::shared_ptr<XObject> KGWrappedSurface<XObject>::GetObject() const
{
    return fObject;
}

template<class XObject> void KGWrappedSurface<XObject>::AreaInitialize() const
{
    fObject->Initialize();
    return;
}

template<class XObject> void KGWrappedSurface<XObject>::AreaAccept(KGVisitor* aVisitor)
{
    auto* tWrappedSurfaceVisitor = dynamic_cast<typename KGWrappedSurface<XObject>::Visitor*>(aVisitor);
    if (tWrappedSurfaceVisitor != nullptr) {
        tWrappedSurfaceVisitor->VisitWrappedSurface(this);
        return;
    }
    KGArea::AreaAccept(aVisitor);
    return;
}

template<class XObject> bool KGWrappedSurface<XObject>::AreaAbove(const katrin::KThreeVector& aQuery) const
{
    if (fObject->ContainsPoint((const double*) (aQuery)) == true) {
        return false;
    }
    return true;
}

template<class XObject>
katrin::KThreeVector KGWrappedSurface<XObject>::AreaPoint(const katrin::KThreeVector& aQuery) const
{
    katrin::KThreeVector tPoint;
    fObject->DistanceTo((const double*) (aQuery), (double*) (tPoint));

    return tPoint;
}

template<class XObject>
katrin::KThreeVector KGWrappedSurface<XObject>::AreaNormal(const katrin::KThreeVector& aQuery) const
{
    katrin::KThreeVector tNormal;
    fObject->DistanceTo((const double*) (aQuery), nullptr, (double*) (tNormal));

    return tNormal;
}
}  // namespace KGeoBag

#endif
