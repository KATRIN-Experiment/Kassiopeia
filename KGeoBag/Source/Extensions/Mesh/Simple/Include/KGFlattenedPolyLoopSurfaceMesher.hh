#ifndef KGeoBag_KGFlattenedPolyLoopSurfaceMesher_hh_
#define KGeoBag_KGFlattenedPolyLoopSurfaceMesher_hh_

#include "KGFlattenedPolyLoopSurface.hh"
#include "KGSimpleMesher.hh"

namespace KGeoBag
{
class KGFlattenedPolyLoopSurfaceMesher : virtual public KGSimpleMesher, public KGFlattenedPolyLoopSurface::Visitor
{
  public:
    using KGMesherBase::VisitExtendedSpace;
    using KGMesherBase::VisitExtendedSurface;

  public:
    KGFlattenedPolyLoopSurfaceMesher();
    ~KGFlattenedPolyLoopSurfaceMesher() override;

  protected:
    void VisitFlattenedClosedPathSurface(KGFlattenedPolyLoopSurface* aFlattenedPolyLoopSurface) override;
};

}  // namespace KGeoBag

#endif
