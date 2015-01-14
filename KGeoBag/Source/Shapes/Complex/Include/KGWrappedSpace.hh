#ifndef KGWRAPPEDSPACE_HH_
#define KGWRAPPEDSPACE_HH_

#include "KGCore.hh"

#include "KGWrappedSurface.hh"

#include <limits>

namespace KGeoBag
{
  template<class XObject>
  class KGWrappedSpace : public KGVolume
  {
  public:
    class Visitor : virtual public KGVisitor
    {
    public:
      Visitor() {}
      virtual ~Visitor() {}

      virtual void VisitWrappedSpace(KGWrappedSpace<XObject>* aWrappedSpace) = 0;
    };

  public:
    KGWrappedSpace();
    KGWrappedSpace(XObject* anObject);
    KGWrappedSpace(const KSmartPointer<XObject>& anObject);
    KGWrappedSpace(const KGWrappedSpace& aCopy);
    virtual ~KGWrappedSpace();

  public:
    void SetObject(KSmartPointer<XObject> anObject);
    KSmartPointer<XObject> GetObject() const;

  public:
    virtual void VolumeInitialize(BoundaryContainer& aBoundaryContainer) const;
    virtual void VolumeAccept(KGVisitor* aVisitor);
    virtual bool VolumeOutside(const KThreeVector& aPoint) const;
    virtual KThreeVector VolumePoint(const KThreeVector& aPoint) const;
    virtual KThreeVector VolumeNormal(const KThreeVector& aPoint) const;

  protected:
    KSmartPointer<XObject> fObject;
  };

  template<class XObject>
  KGWrappedSpace<XObject>::KGWrappedSpace() : KGVolume(), fObject(new XObject)
  {

  }

  template<class XObject>
  KGWrappedSpace<XObject>::KGWrappedSpace(XObject* anObject) : KGVolume(), fObject(anObject)
  {

  }

  template<class XObject>
  KGWrappedSpace<XObject>::KGWrappedSpace(const KSmartPointer<XObject>& anObject) : KGVolume()
  {
    fObject = anObject;
  }

  template<class XObject>
  KGWrappedSpace<XObject>::KGWrappedSpace(const KGWrappedSpace& aCopy) : KGVolume(aCopy)
  {
    fObject = aCopy.fObject;
  }

  template<class XObject>
  KGWrappedSpace<XObject>::~KGWrappedSpace()
  {

  }

  template<class XObject>
  void KGWrappedSpace<XObject>::SetObject(KSmartPointer<XObject> anObject)
  {
    fObject = anObject;
    return;
  }

  template<class XObject>
  KSmartPointer<XObject> KGWrappedSpace<XObject>::GetObject() const
  {
    return fObject;
  }

  template<class XObject>
  void KGWrappedSpace<XObject>::VolumeInitialize(BoundaryContainer& aBoundaryContainer) const
  {
    fObject->Initialize();
    KGWrappedSurface<XObject>* tSurface =
      new KGWrappedSurface<XObject>(fObject);
    aBoundaryContainer.push_back(tSurface);
    return;
  }

  template<class XObject>
  void KGWrappedSpace<XObject>::VolumeAccept(KGVisitor* aVisitor)
  {
    typename KGWrappedSpace<XObject>::Visitor* tWrappedSpaceVisitor = dynamic_cast<typename KGWrappedSpace<XObject>::Visitor*>(aVisitor);
    if(tWrappedSpaceVisitor != NULL)
    {
      tWrappedSpaceVisitor->VisitWrappedSpace(this);
      return;
    }
    KGVolume::VolumeAccept( aVisitor );
    return;
  }

  template<class XObject>
  bool KGWrappedSpace<XObject>::VolumeOutside(const KThreeVector& aQuery) const
  {
    if(fObject->ContainsPoint((const double*) (aQuery)) == true)
    {
      return false;
    }
    return true;
  }

  template<class XObject>
  KThreeVector KGWrappedSpace<XObject>::VolumePoint(const KThreeVector& aQuery) const
  {
    KThreeVector tPoint;
    fObject->DistanceTo((const double*) (aQuery), (double*) (tPoint));

    return tPoint;
  }

  template<class XObject>
  KThreeVector KGWrappedSpace<XObject>::VolumeNormal(const KThreeVector& aQuery) const
  {
    KThreeVector tNormal;
    fObject->DistanceTo((const double*) (aQuery), NULL, (double*) (tNormal));

    return tNormal;
  }
}

#endif
