#ifndef KGeoBag_KGExtrudedCircleSpaceMesher_hh_
#define KGeoBag_KGExtrudedCircleSpaceMesher_hh_

#include "KGExtrudedCircleSpace.hh"
#include "KGSimpleMesher.hh"

namespace KGeoBag
{
class KGExtrudedCircleSpaceMesher : virtual public KGSimpleMesher, public KGExtrudedCircleSpace::Visitor
{
  public:
    using KGMesherBase::VisitExtendedSpace;
    using KGMesherBase::VisitExtendedSurface;

  public:
    KGExtrudedCircleSpaceMesher();
    ~KGExtrudedCircleSpaceMesher() override;

  protected:
    void VisitExtrudedClosedPathSpace(KGExtrudedCircleSpace* aExtrudedCircleSpace) override;
};

}  // namespace KGeoBag

#endif
