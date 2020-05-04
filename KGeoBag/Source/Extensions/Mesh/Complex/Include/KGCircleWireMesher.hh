#ifndef KGeoBag_KGCircleWireMesher_hh_
#define KGeoBag_KGCircleWireMesher_hh_

#include "KGCircleWireSurface.hh"
#include "KGComplexMesher.hh"

namespace KGeoBag
{
class KGCircleWireMesher : virtual public KGComplexMesher, public KGWrappedSurface<KGCircleWire>::Visitor
{
  public:
    using KGMesherBase::VisitExtendedSpace;
    using KGMesherBase::VisitExtendedSurface;

  public:
    KGCircleWireMesher() {}
    ~KGCircleWireMesher() override {}

  protected:
    void VisitWrappedSurface(KGWrappedSurface<KGCircleWire>* circleWireSurface) override;
};

}  // namespace KGeoBag

#endif
