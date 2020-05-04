#ifndef KGVOLUME_HH_
#define KGVOLUME_HH_

#include "KGArea.hh"

#include <memory>
#include <set>

namespace KGeoBag
{

class KGVolume : public KTagged
{
  public:
    class Visitor
    {
      public:
        Visitor() {}
        virtual ~Visitor() {}
        virtual void VisitVolume(KGVolume*) = 0;
    };

  public:
    typedef std::shared_ptr<KGBoundary> BoundaryPointer;
    typedef std::vector<BoundaryPointer> BoundaryContainer;
    typedef BoundaryContainer::iterator BoundaryIt;
    typedef BoundaryContainer::const_iterator BoundaryCIt;

  public:
    KGVolume();
    KGVolume(const KGVolume& aVolume);
    ~KGVolume() override;

  public:
    const BoundaryContainer& Boundaries() const;

  public:
    void Accept(KGVisitor* aVisitor);

  protected:
    virtual void VolumeAccept(KGVisitor* aVisitor);

  public:
    bool Outside(const KThreeVector& aPoint) const;
    KThreeVector Point(const KThreeVector& aPoint) const;
    KThreeVector Normal(const KThreeVector& aPoint) const;

  protected:
    virtual bool VolumeOutside(const KThreeVector& aPoint) const = 0;
    virtual KThreeVector VolumePoint(const KThreeVector& aPoint) const = 0;
    virtual KThreeVector VolumeNormal(const KThreeVector& aPoint) const = 0;

  protected:
    void Check() const;
    virtual void VolumeInitialize(BoundaryContainer& aBoundaryContainer) const = 0;
    mutable bool fInitialized;

  private:
    mutable BoundaryContainer fBoundaries;
};

}  // namespace KGeoBag

#endif
