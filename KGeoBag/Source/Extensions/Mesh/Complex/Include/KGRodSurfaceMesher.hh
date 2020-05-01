#ifndef KGeoBag_KGRodSurfaceMesher_hh_
#define KGeoBag_KGRodSurfaceMesher_hh_

#include "KGComplexMesher.hh"
#include "KGRodSurface.hh"

namespace KGeoBag
{
class KGRodSurfaceMesher : virtual public KGComplexMesher, public KGWrappedSurface<KGRod>::Visitor
{
  public:
    using KGMesherBase::VisitExtendedSpace;
    using KGMesherBase::VisitExtendedSurface;

  public:
    KGRodSurfaceMesher() {}
    ~KGRodSurfaceMesher() override {}

  protected:
    void VisitWrappedSurface(KGWrappedSurface<KGRod>* rodSurface) override;

    void Normalize(const double* p1, const double* p2, double* norm);

    void GetNormal(const double* p1, const double* p2, const double* oldNormal, double* normal);

    void AddTrapezoid(const double* P1, const double* P2, const double* P3, const double* P4, const int nDisc);
};
}  // namespace KGeoBag

#endif /* KGRODSURFACEMESHER_HH_ */
