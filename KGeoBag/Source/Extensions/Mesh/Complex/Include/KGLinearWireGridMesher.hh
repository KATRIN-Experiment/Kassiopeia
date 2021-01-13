#ifndef KGeoBag_KGLinearWireGridMesher_hh_
#define KGeoBag_KGLinearWireGridMesher_hh_

#include "KGComplexMesher.hh"
#include "KGLinearWireGridSurface.hh"

namespace KGeoBag
{
class KGLinearWireGridMesher : virtual public KGComplexMesher, public KGWrappedSurface<KGLinearWireGrid>::Visitor
{
  public:
    using KGMesherBase::VisitExtendedSpace;
    using KGMesherBase::VisitExtendedSurface;

  public:
    KGLinearWireGridMesher() = default;
    ~KGLinearWireGridMesher() override = default;

  protected:
    void VisitWrappedSurface(KGWrappedSurface<KGLinearWireGrid>* linearWireGridSurface) override;
};

}  // namespace KGeoBag

#endif
