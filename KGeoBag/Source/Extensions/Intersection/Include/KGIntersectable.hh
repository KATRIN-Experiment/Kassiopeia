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

    virtual bool Intersection(katrin::KThreeVector& aStart, katrin::KThreeVector& anEnd,
                              katrin::KThreeVector& aResult) const = 0;

  protected:
};

class KGIntersectableSurface
{
  public:
    KGIntersectableSurface();
    virtual ~KGIntersectableSurface();

    void SetIntersector(KGAnalyticIntersector* intersector);
    void SetSurface(const KGSurface& surface);

    bool Intersection(const katrin::KThreeVector& aStart, const katrin::KThreeVector& anEnd,
                      katrin::KThreeVector& aResult) const;

  private:
    static bool NumericIntersection(const katrin::KThreeVector& aLocalStart, const katrin::KThreeVector& aLocalEnd,
                                    katrin::KThreeVector& aLocalResult);

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
