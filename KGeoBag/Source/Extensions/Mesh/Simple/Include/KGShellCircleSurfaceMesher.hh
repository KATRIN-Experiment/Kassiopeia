#ifndef KGeoBag_KGShellCircleSurfaceMesher_hh_
#define KGeoBag_KGShellCircleSurfaceMesher_hh_

#include "KGShellCircleSurface.hh"
#include "KGSimpleMesher.hh"

namespace KGeoBag
{
class KGShellCircleSurfaceMesher : virtual public KGSimpleMesher, public KGShellCircleSurface::Visitor
{
  public:
    using KGMesherBase::VisitExtendedSpace;
    using KGMesherBase::VisitExtendedSurface;

  public:
    KGShellCircleSurfaceMesher();
    ~KGShellCircleSurfaceMesher() override;

  protected:
    void VisitShellPathSurface(KGShellCircleSurface* aShellCircleSurface) override;
};

}  // namespace KGeoBag

#endif
