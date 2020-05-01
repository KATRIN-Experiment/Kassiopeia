#ifndef KGeoBag_KGRotatedPolyLoopSurfaceAxialMesher_hh_
#define KGeoBag_KGRotatedPolyLoopSurfaceAxialMesher_hh_

#include "KGRotatedPolyLoopSurface.hh"
#include "KGSimpleAxialMesher.hh"

namespace KGeoBag
{
class KGRotatedPolyLoopSurfaceAxialMesher : virtual public KGSimpleAxialMesher, public KGRotatedPolyLoopSurface::Visitor
{
  public:
    using KGAxialMesherBase::VisitExtendedSpace;
    using KGAxialMesherBase::VisitExtendedSurface;

  public:
    KGRotatedPolyLoopSurfaceAxialMesher();
    ~KGRotatedPolyLoopSurfaceAxialMesher() override;

  protected:
    void VisitRotatedPathSurface(KGRotatedPolyLoopSurface* aRotatedPolyLoopSurface) override;
};

}  // namespace KGeoBag

#endif
