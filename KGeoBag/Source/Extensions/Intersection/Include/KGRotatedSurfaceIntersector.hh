#ifndef KGROTATEDSURFACEINTERSECTOR_DEF
#define KGROTATEDSURFACEINTERSECTOR_DEF

#include "KGRotatedSurface.hh"

#include "KGIntersectorInitializer.hh"

namespace KGeoBag
{
  class KGRotatedSurfaceIntersectorInitializer :
    public KGRotatedSurface::Visitor,
    virtual public KGIntersectorInitializer
  {
  public:
    KGRotatedSurfaceIntersectorInitializer() {}
    virtual ~KGRotatedSurfaceIntersectorInitializer() {}

  protected:
    void VisitRotatedSurface(const KGRotatedSurface* rotatedSurface);

  };

  class KGRotatedSurfaceIntersector : virtual public KGAnalyticIntersector
  {
  public:
    KGRotatedSurfaceIntersector(const KGRotatedSurface& rotatedSurface);
    virtual ~KGRotatedSurfaceIntersector() {}

    virtual bool Intersection(KThreeVector& aStart,
			      KThreeVector& anEnd,
			      KThreeVector& aResult) const;

  protected:
    const KGRotatedSurface& fRotatedSurface;
  };
}

#endif /* KGROTATEDSURFACEINTERSECTOR_DEF */
