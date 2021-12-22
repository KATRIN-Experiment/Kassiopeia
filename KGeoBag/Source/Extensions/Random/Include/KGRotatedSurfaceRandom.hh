#ifndef KGROTATEDSURFACERANDOM_DEF
#define KGROTATEDSURFACERANDOM_DEF

#include "KGRotatedSurface.hh"
#include "KGShapeRandom.hh"

namespace KGeoBag
{
class KGRotatedSurfaceRandom : virtual public KGShapeRandom, public KGRotatedSurface::Visitor
{
  public:
    KGRotatedSurfaceRandom() : KGShapeRandom() {}
    ~KGRotatedSurfaceRandom() override = default;

    void VisitWrappedSurface(KGRotatedSurface* rotatedSurface) override;

    using KGShapeRandom::Random;

  private:
    katrin::KThreeVector Random(const KGRotatedObject::Line* line);
    katrin::KThreeVector Random(const KGRotatedObject::Arc* arc);
};
}  // namespace KGeoBag

#endif /* KGROTATEDSURFACERANDOM_DEF */
