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
    KGRodSurfaceMesher() = default;
    ~KGRodSurfaceMesher() override = default;

  protected:
    void VisitWrappedSurface(KGWrappedSurface<KGRod>* rodSurface) override;

    static void Normalize(const double* p1, const double* p2, double* norm);

    static void GetNormal(const double* p1, const double* p2, const double* oldNormal, double* normal);

    void AddTrapezoid(const double* P1, const double* P2, const double* P3, const double* P4, const int nDisc);
};
}  // namespace KGeoBag

#endif /* KGRODSURFACEMESHER_HH_ */
