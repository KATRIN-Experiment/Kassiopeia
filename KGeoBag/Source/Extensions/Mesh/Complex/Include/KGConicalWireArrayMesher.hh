#ifndef KGeoBag_KGConicalWireArrayMesher_hh_
#define KGeoBag_KGConicalWireArrayMesher_hh_

#include "KGComplexMesher.hh"
#include "KGConicalWireArraySurface.hh"

namespace KGeoBag
{
class KGConicalWireArrayMesher : virtual public KGComplexMesher, public KGWrappedSurface<KGConicalWireArray>::Visitor
{
  public:
    using KGMesherBase::VisitExtendedSpace;
    using KGMesherBase::VisitExtendedSurface;

  public:
    KGConicalWireArrayMesher() {}
    ~KGConicalWireArrayMesher() override {}

  protected:
    void VisitWrappedSurface(KGWrappedSurface<KGConicalWireArray>* conicalWireArraySurface) override;
};

}  // namespace KGeoBag

#endif
