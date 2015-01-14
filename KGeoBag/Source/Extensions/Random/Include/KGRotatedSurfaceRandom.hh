#ifndef KGROTATEDSURFACERANDOM_DEF
#define KGROTATEDSURFACERANDOM_DEF

#include "KGRotatedSurface.hh"

#include "KGShapeRandom.hh"

namespace KGeoBag
{
  class KGRotatedSurfaceRandom : virtual public KGShapeRandom,
				 public KGRotatedSurface::Visitor
  {
  public:
    KGRotatedSurfaceRandom() : KGShapeRandom() {}
    virtual ~KGRotatedSurfaceRandom() {}

    void VisitWrappedSurface(KGRotatedSurface* rotatedSurface);

    using KGShapeRandom::Random;

  private:
    KThreeVector Random(const KGRotatedObject::Line* line);
    KThreeVector Random(const KGRotatedObject::Arc* arc);
  };
}

#endif /* KGROTATEDSURFACERANDOM_DEF */
