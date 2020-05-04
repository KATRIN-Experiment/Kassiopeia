#ifndef KGeoBag_KGFlattenedCircleSurfaceMesher_hh_
#define KGeoBag_KGFlattenedCircleSurfaceMesher_hh_

#include "KGFlattenedCircleSurface.hh"
#include "KGSimpleMesher.hh"

namespace KGeoBag
{
class KGFlattenedCircleSurfaceMesher : virtual public KGSimpleMesher, public KGFlattenedCircleSurface::Visitor
{
  public:
    using KGMesherBase::VisitExtendedSpace;
    using KGMesherBase::VisitExtendedSurface;

  public:
    KGFlattenedCircleSurfaceMesher();
    ~KGFlattenedCircleSurfaceMesher() override;

  protected:
    void VisitFlattenedClosedPathSurface(KGFlattenedCircleSurface* aFlattenedCircleSurface) override;
};

}  // namespace KGeoBag

#endif
