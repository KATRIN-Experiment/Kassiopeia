#ifndef KGeoBag_KGShellArcSegmentSurfaceMesher_hh_
#define KGeoBag_KGShellArcSegmentSurfaceMesher_hh_

#include "KGShellArcSegmentSurface.hh"
#include "KGSimpleMesher.hh"

namespace KGeoBag
{
class KGShellArcSegmentSurfaceMesher : virtual public KGSimpleMesher, public KGShellArcSegmentSurface::Visitor
{
  public:
    using KGMesherBase::VisitExtendedSpace;
    using KGMesherBase::VisitExtendedSurface;

  public:
    KGShellArcSegmentSurfaceMesher();
    ~KGShellArcSegmentSurfaceMesher() override;

  protected:
    void VisitShellPathSurface(KGShellArcSegmentSurface* aShellArcSegmentSurface) override;
};

}  // namespace KGeoBag

#endif
