#ifndef KGINTERSECTABLE_HH_
#define KGINTERSECTABLE_HH_

#include "KGCore.hh"

namespace KGeoBag
{
class KGAnalyticIntersector
{
  public:
    KGAnalyticIntersector() = default;
    virtual ~KGAnalyticIntersector() = default;

    virtual bool Intersection(KThreeVector& aStart, KGeoBag::KThreeVector& anEnd,
                              KGeoBag::KThreeVector& aResult) const = 0;

  protected:
};

class KGIntersectableSurface
{
  public:
    KGIntersectableSurface();
    virtual ~KGIntersectableSurface();

    void SetIntersector(KGAnalyticIntersector* intersector);
    void SetSurface(const KGSurface& surface);

    bool Intersection(const KGeoBag::KThreeVector& aStart, const KGeoBag::KThreeVector& anEnd,
                      KGeoBag::KThreeVector& aResult) const;

  private:
    static bool NumericIntersection(const KGeoBag::KThreeVector& aLocalStart, const KGeoBag::KThreeVector& aLocalEnd,
                                    KGeoBag::KThreeVector& aLocalResult);

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
