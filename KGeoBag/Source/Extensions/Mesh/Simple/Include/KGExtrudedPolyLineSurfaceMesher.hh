#ifndef KGeoBag_KGExtrudedPolyLineSurfaceMesher_hh_
#define KGeoBag_KGExtrudedPolyLineSurfaceMesher_hh_

#include "KGExtrudedPolyLineSurface.hh"
#include "KGSimpleMesher.hh"

namespace KGeoBag
{
class KGExtrudedPolyLineSurfaceMesher : virtual public KGSimpleMesher, public KGExtrudedPolyLineSurface::Visitor
{
  public:
    using KGMesherBase::VisitExtendedSpace;
    using KGMesherBase::VisitExtendedSurface;

  public:
    KGExtrudedPolyLineSurfaceMesher();
    ~KGExtrudedPolyLineSurfaceMesher() override;

  protected:
    void VisitExtrudedPathSurface(KGExtrudedPolyLineSurface* aExtrudedPolyLineSurface) override;
};

}  // namespace KGeoBag

#endif
