#ifndef KGVOLUME_HH_
#define KGVOLUME_HH_

#include "KGArea.hh"

#include <memory>
#include <set>

namespace KGeoBag
{

class KGVolume : public katrin::KTagged
{
  public:
    class Visitor
    {
      public:
        Visitor() = default;
        virtual ~Visitor() = default;
        virtual void VisitVolume(KGVolume*) = 0;
    };

  public:
    typedef std::shared_ptr<KGBoundary> BoundaryPointer;
    using BoundaryContainer = std::vector<BoundaryPointer>;
    using BoundaryIt = BoundaryContainer::iterator;
    using BoundaryCIt = BoundaryContainer::const_iterator;

  public:
    KGVolume();
    KGVolume(const KGVolume& aVolume);
    ~KGVolume() override;

    static std::string Name()
    {
        return "volume";
    }

  public:
    const BoundaryContainer& Boundaries() const;

  public:
    void Accept(KGVisitor* aVisitor);

  protected:
    virtual void VolumeAccept(KGVisitor* aVisitor);

  public:
    bool Outside(const KGeoBag::KThreeVector& aPoint) const;
    KGeoBag::KThreeVector Point(const KGeoBag::KThreeVector& aPoint) const;
    KGeoBag::KThreeVector Normal(const KGeoBag::KThreeVector& aPoint) const;

  protected:
    virtual bool VolumeOutside(const KGeoBag::KThreeVector& aPoint) const = 0;
    virtual KGeoBag::KThreeVector VolumePoint(const KGeoBag::KThreeVector& aPoint) const = 0;
    virtual KGeoBag::KThreeVector VolumeNormal(const KGeoBag::KThreeVector& aPoint) const = 0;

  protected:
    void Check() const;
    virtual void VolumeInitialize(BoundaryContainer& aBoundaryContainer) const = 0;
    mutable bool fInitialized;

  private:
    mutable BoundaryContainer fBoundaries;
};

}  // namespace KGeoBag

#endif
