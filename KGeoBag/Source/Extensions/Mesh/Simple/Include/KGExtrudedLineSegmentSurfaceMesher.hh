#ifndef KGeoBag_KGExtrudedLineSegmentSurfaceMesher_hh_
#define KGeoBag_KGExtrudedLineSegmentSurfaceMesher_hh_

#include "KGExtrudedLineSegmentSurface.hh"
#include "KGSimpleMesher.hh"

namespace KGeoBag
{
class KGExtrudedLineSegmentSurfaceMesher : virtual public KGSimpleMesher, public KGExtrudedLineSegmentSurface::Visitor
{
  public:
    using KGMesherBase::VisitExtendedSpace;
    using KGMesherBase::VisitExtendedSurface;

  public:
    KGExtrudedLineSegmentSurfaceMesher();
    ~KGExtrudedLineSegmentSurfaceMesher() override;

  protected:
    void VisitExtrudedPathSurface(KGExtrudedLineSegmentSurface* aExtrudedLineSegmentSurface) override;
};

}  // namespace KGeoBag

#endif
