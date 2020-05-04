#ifndef KGeoBag_KGExtrudedPolyLoopSurfaceMesher_hh_
#define KGeoBag_KGExtrudedPolyLoopSurfaceMesher_hh_

#include "KGExtrudedPolyLoopSurface.hh"
#include "KGSimpleMesher.hh"

namespace KGeoBag
{
class KGExtrudedPolyLoopSurfaceMesher : virtual public KGSimpleMesher, public KGExtrudedPolyLoopSurface::Visitor
{
  public:
    using KGMesherBase::VisitExtendedSpace;
    using KGMesherBase::VisitExtendedSurface;

  public:
    KGExtrudedPolyLoopSurfaceMesher();
    ~KGExtrudedPolyLoopSurfaceMesher() override;

  protected:
    void VisitExtrudedPathSurface(KGExtrudedPolyLoopSurface* aExtrudedPolyLoopSurface) override;
};

}  // namespace KGeoBag

#endif
