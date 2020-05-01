#ifndef KGeoBag_KGShellPolyLoopSurfaceMesher_hh_
#define KGeoBag_KGShellPolyLoopSurfaceMesher_hh_

#include "KGShellPolyLoopSurface.hh"
#include "KGSimpleMesher.hh"

namespace KGeoBag
{
class KGShellPolyLoopSurfaceMesher : virtual public KGSimpleMesher, public KGShellPolyLoopSurface::Visitor
{
  public:
    using KGMesherBase::VisitExtendedSpace;
    using KGMesherBase::VisitExtendedSurface;

  public:
    KGShellPolyLoopSurfaceMesher();
    ~KGShellPolyLoopSurfaceMesher() override;

  protected:
    void VisitShellPathSurface(KGShellPolyLoopSurface* aShellPolyLoopSurface) override;
};

}  // namespace KGeoBag

#endif
