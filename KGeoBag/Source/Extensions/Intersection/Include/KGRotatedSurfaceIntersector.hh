#ifndef KGROTATEDSURFACEINTERSECTOR_DEF
#define KGROTATEDSURFACEINTERSECTOR_DEF

#include "KGIntersectorInitializer.hh"
#include "KGRotatedSurface.hh"

namespace KGeoBag
{
class KGRotatedSurfaceIntersectorInitializer : public KGRotatedSurface::Visitor, virtual public KGIntersectorInitializer
{
  public:
    KGRotatedSurfaceIntersectorInitializer() = default;
    ~KGRotatedSurfaceIntersectorInitializer() override = default;

  protected:
    void VisitRotatedSurface(const KGRotatedSurface* rotatedSurface);
};

class KGRotatedSurfaceIntersector : virtual public KGAnalyticIntersector
{
  public:
    KGRotatedSurfaceIntersector(const KGRotatedSurface& rotatedSurface);
    ~KGRotatedSurfaceIntersector() override = default;

    bool Intersection(katrin::KThreeVector& aStart, katrin::KThreeVector& anEnd,
                      katrin::KThreeVector& aResult) const override;

  protected:
    const KGRotatedSurface& fRotatedSurface;
};
}  // namespace KGeoBag

#endif /* KGROTATEDSURFACEINTERSECTOR_DEF */
