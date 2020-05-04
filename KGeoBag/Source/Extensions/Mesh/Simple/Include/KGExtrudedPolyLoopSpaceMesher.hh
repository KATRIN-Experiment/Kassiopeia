#ifndef KGeoBag_KGExtrudedPolyLoopSpaceMesher_hh_
#define KGeoBag_KGExtrudedPolyLoopSpaceMesher_hh_

#include "KGExtrudedPolyLoopSpace.hh"
#include "KGSimpleMesher.hh"

namespace KGeoBag
{
class KGExtrudedPolyLoopSpaceMesher : virtual public KGSimpleMesher, public KGExtrudedPolyLoopSpace::Visitor
{
  public:
    using KGMesherBase::VisitExtendedSpace;
    using KGMesherBase::VisitExtendedSurface;

  public:
    KGExtrudedPolyLoopSpaceMesher();
    ~KGExtrudedPolyLoopSpaceMesher() override;

  protected:
    void VisitExtrudedClosedPathSpace(KGExtrudedPolyLoopSpace* aExtrudedPolyLoopSpace) override;
};

}  // namespace KGeoBag

#endif
