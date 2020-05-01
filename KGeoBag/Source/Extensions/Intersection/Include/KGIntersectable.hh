#ifndef KGINTERSECTABLE_HH_
#define KGINTERSECTABLE_HH_

#include "KGCore.hh"

namespace KGeoBag
{
class KGAnalyticIntersector
{
  public:
    KGAnalyticIntersector() {}
    virtual ~KGAnalyticIntersector() {}

    virtual bool Intersection(KThreeVector& aStart, KThreeVector& anEnd, KThreeVector& aResult) const = 0;

  protected:
};

class KGIntersectableSurface
{
  public:
    KGIntersectableSurface();
    virtual ~KGIntersectableSurface();

    void SetIntersector(KGAnalyticIntersector* intersector);
    void SetSurface(const KGSurface& surface);

    bool Intersection(const KThreeVector& aStart, const KThreeVector& anEnd, KThreeVector& aResult) const;

  private:
    bool NumericIntersection(const KThreeVector& aLocalStart, const KThreeVector& aLocalEnd,
                             KThreeVector& aLocalResult) const;

    const KGSurface* fSurface;

    KGAnalyticIntersector* fIntersector;
};

class KGIntersectable
{
  public:
    typedef KGIntersectableSurface Surface;
};

}  // namespace KGeoBag

#endif
