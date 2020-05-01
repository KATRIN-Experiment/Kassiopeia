#ifndef KGeoBag_KGRotatedPolyLoopSpaceMesher_hh_
#define KGeoBag_KGRotatedPolyLoopSpaceMesher_hh_

#include "KGRotatedPolyLoopSpace.hh"
#include "KGSimpleMesher.hh"

namespace KGeoBag
{
class KGRotatedPolyLoopSpaceMesher : virtual public KGSimpleMesher, public KGRotatedPolyLoopSpace::Visitor
{
  public:
    using KGMesherBase::VisitExtendedSpace;
    using KGMesherBase::VisitExtendedSurface;

  public:
    KGRotatedPolyLoopSpaceMesher();
    ~KGRotatedPolyLoopSpaceMesher() override;

  protected:
    void VisitRotatedClosedPathSpace(KGRotatedPolyLoopSpace* aRotatedPolyLoopSpace) override;
};

}  // namespace KGeoBag

#endif
