#ifndef KGeoBag_KGRotatedSurfaceMesher_hh_
#define KGeoBag_KGRotatedSurfaceMesher_hh_

#include "KGComplexMesher.hh"
#include "KGRotatedSurface.hh"

namespace KGeoBag
{
class KGRotatedSurfaceMesher : virtual public KGComplexMesher, public KGRotatedSurface::Visitor
{
  public:
    using KGMesherBase::VisitExtendedSpace;
    using KGMesherBase::VisitExtendedSurface;

  public:
    KGRotatedSurfaceMesher() {}
    ~KGRotatedSurfaceMesher() override {}

  protected:
    void VisitWrappedSurface(KGRotatedSurface* rotatedSurface) override;

    void DiscretizeSegment(const KGRotatedObject::Line* line, const unsigned int nPolyBegin,
                           const unsigned int nPolyEnd);
    void DiscretizeSegment(const KGRotatedObject::Arc* arc, const unsigned int nPolyBegin, const unsigned int nPolyEnd);
};
}  // namespace KGeoBag

#endif /* KGROTATEDSURFACEDISCRETIZER_DEF */
