#ifndef KGeoBag_KGBeamSurfaceMesher_hh_
#define KGeoBag_KGBeamSurfaceMesher_hh_

#include "KGBeamSurface.hh"
#include "KGComplexMesher.hh"

namespace KGeoBag
{
class KGBeamSurfaceMesher : virtual public KGComplexMesher, public KGBeamSurface::Visitor
{
  public:
    using KGMesherBase::VisitExtendedSpace;
    using KGMesherBase::VisitExtendedSurface;

  public:
    KGBeamSurfaceMesher() = default;
    ~KGBeamSurfaceMesher() override = default;

  protected:
    void VisitWrappedSurface(KGBeamSurface* beamSurface) override;
};
}  // namespace KGeoBag

#endif /* KGBEAMSURFACEMESHER_HH_ */
