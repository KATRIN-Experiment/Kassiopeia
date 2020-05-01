#ifndef KGeoBag_KGRotatedArcSegmentSpaceMesher_hh_
#define KGeoBag_KGRotatedArcSegmentSpaceMesher_hh_

#include "KGRotatedArcSegmentSpace.hh"
#include "KGSimpleMesher.hh"

namespace KGeoBag
{
class KGRotatedArcSegmentSpaceMesher : virtual public KGSimpleMesher, public KGRotatedArcSegmentSpace::Visitor
{
  public:
    using KGMesherBase::VisitExtendedSpace;
    using KGMesherBase::VisitExtendedSurface;

  public:
    KGRotatedArcSegmentSpaceMesher();
    ~KGRotatedArcSegmentSpaceMesher() override;

  protected:
    void VisitRotatedOpenPathSpace(KGRotatedArcSegmentSpace* aRotatedArcSegmentSpace) override;
};

}  // namespace KGeoBag

#endif
