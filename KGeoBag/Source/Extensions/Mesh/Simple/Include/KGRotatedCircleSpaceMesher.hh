#ifndef KGeoBag_KGRotatedCircleSpaceMesher_hh_
#define KGeoBag_KGRotatedCircleSpaceMesher_hh_

#include "KGRotatedCircleSpace.hh"
#include "KGSimpleMesher.hh"

namespace KGeoBag
{
class KGRotatedCircleSpaceMesher : virtual public KGSimpleMesher, public KGRotatedCircleSpace::Visitor
{
  public:
    using KGMesherBase::VisitExtendedSpace;
    using KGMesherBase::VisitExtendedSurface;

  public:
    KGRotatedCircleSpaceMesher();
    ~KGRotatedCircleSpaceMesher() override;

  protected:
    void VisitRotatedClosedPathSpace(KGRotatedCircleSpace* aRotatedCircleSpace) override;
};

}  // namespace KGeoBag

#endif
