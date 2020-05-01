#ifndef KGeoBag_KGShellLineSegmentSurfaceMesher_hh_
#define KGeoBag_KGShellLineSegmentSurfaceMesher_hh_

#include "KGShellLineSegmentSurface.hh"
#include "KGSimpleMesher.hh"

namespace KGeoBag
{
class KGShellLineSegmentSurfaceMesher : virtual public KGSimpleMesher, public KGShellLineSegmentSurface::Visitor
{
  public:
    using KGMesherBase::VisitExtendedSpace;
    using KGMesherBase::VisitExtendedSurface;

  public:
    KGShellLineSegmentSurfaceMesher();
    ~KGShellLineSegmentSurfaceMesher() override;

  protected:
    void VisitShellPathSurface(KGShellLineSegmentSurface* aShellLineSegmentSurface) override;
};

}  // namespace KGeoBag

#endif
