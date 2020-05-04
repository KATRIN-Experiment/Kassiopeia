#ifndef KGeoBag_KGRotatedArcSegmentSurfaceMesher_hh_
#define KGeoBag_KGRotatedArcSegmentSurfaceMesher_hh_

#include "KGRotatedArcSegmentSurface.hh"
#include "KGSimpleMesher.hh"

namespace KGeoBag
{
class KGRotatedArcSegmentSurfaceMesher : virtual public KGSimpleMesher, public KGRotatedArcSegmentSurface::Visitor
{
  public:
    using KGMesherBase::VisitExtendedSpace;
    using KGMesherBase::VisitExtendedSurface;

  public:
    KGRotatedArcSegmentSurfaceMesher();
    ~KGRotatedArcSegmentSurfaceMesher() override;

  protected:
    void VisitRotatedPathSurface(KGRotatedArcSegmentSurface* aRotatedArcSegmentSurface) override;
};

}  // namespace KGeoBag

#endif
