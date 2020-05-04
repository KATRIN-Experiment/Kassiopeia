#ifndef KGeoBag_KGShellPolyLineSurfaceMesher_hh_
#define KGeoBag_KGShellPolyLineSurfaceMesher_hh_

#include "KGShellPolyLineSurface.hh"
#include "KGSimpleMesher.hh"

namespace KGeoBag
{
class KGShellPolyLineSurfaceMesher : virtual public KGSimpleMesher, public KGShellPolyLineSurface::Visitor
{
  public:
    using KGMesherBase::VisitExtendedSpace;
    using KGMesherBase::VisitExtendedSurface;

  public:
    KGShellPolyLineSurfaceMesher();
    ~KGShellPolyLineSurfaceMesher() override;

  protected:
    void VisitShellPathSurface(KGShellPolyLineSurface* aShellPolyLineSurface) override;
};

}  // namespace KGeoBag

#endif
