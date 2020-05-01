#ifndef KGeoBag_KGExtrudedArcSegmentSurfaceMesher_hh_
#define KGeoBag_KGExtrudedArcSegmentSurfaceMesher_hh_

#include "KGExtrudedArcSegmentSurface.hh"
#include "KGSimpleMesher.hh"

namespace KGeoBag
{
class KGExtrudedArcSegmentSurfaceMesher : virtual public KGSimpleMesher, public KGExtrudedArcSegmentSurface::Visitor
{
  public:
    using KGMesherBase::VisitExtendedSpace;
    using KGMesherBase::VisitExtendedSurface;

  public:
    KGExtrudedArcSegmentSurfaceMesher();
    ~KGExtrudedArcSegmentSurfaceMesher() override;

  protected:
    void VisitExtrudedPathSurface(KGExtrudedArcSegmentSurface* aExtrudedArcSegmentSurface) override;
};

}  // namespace KGeoBag

#endif
