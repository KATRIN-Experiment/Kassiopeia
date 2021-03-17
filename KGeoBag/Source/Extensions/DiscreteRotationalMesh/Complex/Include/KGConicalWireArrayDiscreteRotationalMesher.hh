#ifndef KGeoBag_KGConicalWireArrayDiscreteRotationalMesher_hh_
#define KGeoBag_KGConicalWireArrayDiscreteRotationalMesher_hh_

#include "KGConicalWireArraySurface.hh"
#include "KGDiscreteRotationalMesherBase.hh"

namespace KGeoBag
{
class KGConicalWireArrayDiscreteRotationalMesher :
    virtual public KGDiscreteRotationalMesherBase,
    public KGWrappedSurface<KGConicalWireArray>::Visitor
{
  public:
    KGConicalWireArrayDiscreteRotationalMesher() = default;
    ~KGConicalWireArrayDiscreteRotationalMesher() override = default;

  public:
    void VisitWrappedSurface(KGWrappedSurface<KGConicalWireArray>* conicalWireArraySurface) override;
};

}  // namespace KGeoBag

#endif
