#ifndef KGVOLUME_HH_
#define KGVOLUME_HH_

#include "KGArea.hh"

#include "KThreeVector.hh"

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
    KGVolume& operator=(const KGVolume&);
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
    bool Outside(const katrin::KThreeVector& aPoint) const;
    katrin::KThreeVector Point(const katrin::KThreeVector& aPoint) const;
    katrin::KThreeVector Normal(const katrin::KThreeVector& aPoint) const;

  protected:
    virtual bool VolumeOutside(const katrin::KThreeVector& aPoint) const = 0;
    virtual katrin::KThreeVector VolumePoint(const katrin::KThreeVector& aPoint) const = 0;
    virtual katrin::KThreeVector VolumeNormal(const katrin::KThreeVector& aPoint) const = 0;

  protected:
    void Check() const;
    virtual void VolumeInitialize(BoundaryContainer& aBoundaryContainer) const = 0;
    mutable bool fInitialized;

  private:
    mutable BoundaryContainer fBoundaries;
};

}  // namespace KGeoBag

#endif
