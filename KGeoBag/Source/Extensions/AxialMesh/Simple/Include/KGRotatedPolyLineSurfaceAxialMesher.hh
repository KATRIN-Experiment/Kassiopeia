#ifndef KGeoBag_KGRotatedPolyLineSurfaceAxialMesher_hh_
#define KGeoBag_KGRotatedPolyLineSurfaceAxialMesher_hh_

#include "KGRotatedPolyLineSurface.hh"
#include "KGSimpleAxialMesher.hh"

namespace KGeoBag
{
class KGRotatedPolyLineSurfaceAxialMesher : virtual public KGSimpleAxialMesher, public KGRotatedPolyLineSurface::Visitor
{
  public:
    using KGAxialMesherBase::VisitExtendedSpace;
    using KGAxialMesherBase::VisitExtendedSurface;

  public:
    KGRotatedPolyLineSurfaceAxialMesher();
    ~KGRotatedPolyLineSurfaceAxialMesher() override;

  protected:
    void VisitRotatedPathSurface(KGRotatedPolyLineSurface* aRotatedPolyLineSurface) override;
};

}  // namespace KGeoBag

#endif
