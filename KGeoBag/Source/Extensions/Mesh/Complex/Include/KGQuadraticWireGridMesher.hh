#ifndef KGeoBag_KGQuadraticWireGridMesher_hh_
#define KGeoBag_KGQuadraticWireGridMesher_hh_

#include "KGComplexMesher.hh"
#include "KGQuadraticWireGridSurface.hh"

namespace KGeoBag
{
class KGQuadraticWireGridMesher : virtual public KGComplexMesher, public KGWrappedSurface<KGQuadraticWireGrid>::Visitor
{
  public:
    using KGMesherBase::VisitExtendedSpace;
    using KGMesherBase::VisitExtendedSurface;

  public:
    KGQuadraticWireGridMesher() {}
    ~KGQuadraticWireGridMesher() override {}

  protected:
    void VisitWrappedSurface(KGWrappedSurface<KGQuadraticWireGrid>* quadraticWireGridSurface) override;
};

}  // namespace KGeoBag

#endif
